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

from typing import Any, Callable, TypeVar
import warnings

import numpy as np

import jax
from jax import lax
import jax.numpy as jnp

from jax._src import dtypes
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_zip, unzip2, HashablePartial

zip = safe_zip

T = TypeVar('T')

def ravel_pytree(pytree: T) -> tuple[jax.Array, Callable[[jax.Array], T]]:
  """Ravel (flatten) a pytree of arrays down to a 1D array.

  Args:
    pytree: a pytree of arrays and scalars to ravel.

  Returns:
    A pair where the first element is a 1D array representing the flattened and
    concatenated leaf values, with dtype determined by promoting the dtypes of
    leaf values, and the second element is a callable for unflattening a 1D
    vector of the same length back to a pytree of the same structure as the
    input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
    a convention a 1D empty array of dtype float32 is returned in the first
    component of the output.

  For details on dtype promotion, see
  https://jax.readthedocs.io/en/latest/type_promotion.html.

  Unlike ``jax.tree_util.tree_flatten``, this function flattens all the way down
  to a single 1D ``jax.Array`` representing the flattened and concatenated values of
  all leaves. That is, it flattens not just the pytree structure but also the
  ``jax.Array`` leaves themselves. In contrast, ``jax.tree_util.tree_flatten`` only
  flattens away the tree structure, returning a list of ``jax.Array``s (and a
  ``PyTreeDef`` rather than an unflattener callable).

  For example:

  >>> import jax.numpy as jnp
  >>> from jax.flatten_util import ravel_pytree
  >>> x = [jnp.arange(3.), {'a': jnp.arange(4), 'b': 5}]
  >>> x_flat, unflattener = ravel_pytree(x)
  >>> print(x_flat)
  Array([0., 1., 2., 0., 1., 2., 3., 5.], dtype=float32)
  >>> unflattener(x_flat)
  [Array([0., 1., 2.], dtype=float32), {'a': Array([0, 1, 2, 3], dtype=int32), 'b': Array(5, dtype=int32)}]
  """
  leaves, treedef = tree_flatten(pytree)
  flat, unravel_list = _ravel_list(leaves)
  return flat, HashablePartial(unravel_pytree, treedef, unravel_list)

def unravel_pytree(treedef, unravel_list, flat):
  return tree_unflatten(treedef, unravel_list(flat))

def _ravel_list(lst):
  if not lst: return jnp.array([], jnp.float32), lambda _: []
  from_dtypes = tuple(dtypes.dtype(l) for l in lst)
  to_dtype = dtypes.result_type(*from_dtypes)
  sizes, shapes = unzip2((jnp.size(x), jnp.shape(x)) for x in lst)
  indices = tuple(np.cumsum(sizes))

  if all(dt == to_dtype for dt in from_dtypes):
    # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
    # See https://github.com/jax-ml/jax/issues/7809.
    del from_dtypes, to_dtype
    raveled = jnp.concatenate([jnp.ravel(e) for e in lst])
    return raveled, HashablePartial(_unravel_list_single_dtype, indices, shapes)

  # When there is more than one distinct input dtype, we perform type
  # conversions and produce a dtype-specific unravel function.
  ravel = lambda e: jnp.ravel(lax.convert_element_type(e, to_dtype))
  raveled = jnp.concatenate([ravel(e) for e in lst])
  unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)
  return raveled, unrav

def _unravel_list_single_dtype(indices, shapes, arr):
  chunks = jnp.split(arr, indices[:-1])
  return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]

def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
  arr_dtype = dtypes.dtype(arr)
  if arr_dtype != to_dtype:
    raise TypeError(f"unravel function given array of dtype {arr_dtype}, "
                    f"but expected dtype {to_dtype}")
  chunks = jnp.split(arr, indices[:-1])
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
    return [lax.convert_element_type(chunk.reshape(shape), dtype)
            for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)]
