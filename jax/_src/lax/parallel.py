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
"""
Parallelization primitives.
"""

import string
import warnings
from typing import Union

import numpy as np

from jax import core
from jax._src import dtypes
from jax import tree_util
from . import lax
from jax.core import ShapedArray, AxisName, raise_to_shaped
from jax.interpreters import ad
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.interpreters import batching
from jax._src.util import partial, unzip2, prod, canonicalize_axis, safe_map, moveaxis
from jax.lib import xla_client as xc
from jax.lib import xla_bridge as xb
from jax._src.numpy import lax_numpy

xops = xc.ops

unsafe_map, map = map, safe_map  # type: ignore


### parallel traceables

def psum(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce sum on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Inputs of boolean dtype are converted to integers before the reduction.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform psums over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once.


  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce sum along the axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [6 6 6 6]
  >>> y = jax.pmap(lambda x: x / jax.lax.psum(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [0.         0.16666667 0.33333334 0.5       ]
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  leaves = [lax.convert_element_type(l, np.int32)
            if dtypes.dtype(l) == np.bool_ else l for l in leaves]
  out_flat = psum_p.bind(*leaves, axes=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

def pmean(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce mean on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmeans over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce mean along the axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [1.5 1.5 1.5 1.5]
  >>> y = jax.pmap(lambda x: x / jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [0.        0.6666667 1.3333334 2.       ]
  """
  x = psum(x, axis_name=axis_name, axis_index_groups=axis_index_groups)
  n = psum(1, axis_name=axis_name, axis_index_groups=axis_index_groups)
  return tree_util.tree_map(lambda v: v / n, x)

def pmax(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce max on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmaxes over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce max along the axis ``axis_name``.
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  out_flat = pmax_p.bind(*leaves, axes=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

def pmin(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce min on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmins over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce min along the axis ``axis_name``.
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  out_flat = pmin_p.bind(*leaves, axes=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

# TODO(mattjj): add a pargmin_p, or add named axis support to lax.argmin_p
def pargmin(x, axis_name):
  if isinstance(axis_name, (tuple, list)):
    raise TypeError(f"pargmin only accepts a single axis, got {axis_name}")
  return _axis_index_of_val(x, pmin(x, axis_name), axis_name)

# TODO(mattjj): add a pargmax_p, or add named axis support to lax.argmax_p
def pargmax(x, axis_name):
  if isinstance(axis_name, (tuple, list)):
    raise TypeError(f"pargmin only accepts a single axis, got {axis_name}")
  return _axis_index_of_val(x, pmax(x, axis_name), axis_name)

def _axis_index_of_val(x, val, axis_name):
  idx = axis_index(axis_name)
  validx = lax_numpy.where(val == x, idx, dtypes.iinfo(dtypes.dtype(idx)).max)
  return pmin(validx, axis_name)

def _validate_reduce_axis_index_groups(axis_index_groups):
  if axis_index_groups is None:
    return
  axis_space = range(sum(len(group) for group in axis_index_groups))
  if {i for g in axis_index_groups for i in g} != set(axis_space):
    raise ValueError("axis_index_groups must cover all indices exactly once")

def ppermute(x, axis_name, perm):
  """Perform a collective permutation according to the permutation ``perm``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This function is an analog of the CollectivePermute XLA HLO.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    perm: list of pairs of ints, representing
      ``(source_index, destination_index)``
      pairs that encode how the mapped axis named ``axis_name`` should be
      shuffled. The integer values are treated as indices into the mapped axis
      ``axis_name``. Any two pairs should not have the same source index or the
      same destination index. For each index of the axis ``axis_name`` that does
      not correspond to a destination index in ``perm``, the corresponding
      values in the result are filled with zeros of the appropriate type.

  Returns:
    Array(s) with the same shape as ``x`` with slices along the axis
    ``axis_name`` gathered from ``x`` according to the permutation ``perm``.
  """
  return tree_util.tree_map(
      partial(ppermute_p.bind, axis_name=axis_name, perm=tuple(perm)), x)

def pshuffle(x, axis_name, perm):
  """Convenience wrapper of jax.lax.ppermute with alternate permutation encoding

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    perm: list of of ints encoding sources for the permutation to be applied to
      the axis named ``axis_name``, so that the output at axis index i
      comes from the input at axis index perm[i]. Every integer in [0, N) should
      be included exactly once for axis size N.

  Returns:
    Array(s) with the same shape as ``x`` with slices along the axis
    ``axis_name`` gathered from ``x`` according to the permutation ``perm``.
  """
  if set(perm) != set(range(len(perm))):
    raise ValueError(f"`perm` does not represent a permutation: {perm}")
  return ppermute(x, axis_name, list(zip(perm, range(len(perm)))))


def pswapaxes(x, axis_name, axis, *, axis_index_groups=None):
  """Swap the pmapped axis ``axis_name`` with the unmapped axis ``axis``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  The group size of the mapped axis size must be equal to the size of the
  unmapped axis; that is, we must have
  ``lax.psum(1, axis_name, axis_index_groups=axis_index_groups) == x.shape[axis]``.
  By default, when ``axis_index_groups=None``, this encompasses all the devices.

  This function is a special case of ``all_to_all`` where the pmapped axis of
  the input is placed at the position ``axis`` in the output. That is, it is
  equivalent to ``all_to_all(x, axis_name, axis, axis)``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run pswapaxes over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x``.
  """
  return all_to_all(x, axis_name, axis, axis, axis_index_groups=axis_index_groups)

def all_to_all(x, axis_name, split_axis, concat_axis, *, axis_index_groups=None, tiled=False):
  """Materialize the mapped axis and map a different axis.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  In the output, the input mapped axis ``axis_name`` is materialized at the
  logical axis position ``concat_axis``, and the input unmapped axis at position
  ``split_axis`` is mapped with the name ``axis_name``.

  The group size of the mapped axis size must be equal to the size of the
  unmapped axis; that is, we must have
  ``lax.psum(1, axis_name, axis_index_groups=axis_index_groups) == x.shape[axis]``.
  By default, when ``axis_index_groups=None``, this encompasses all the devices.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    split_axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    concat_axis: int indicating the position in the output to materialize the
      mapped axis of the input with the name ``axis_name``.
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run all_to_all over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.
    tiled: when True, all_to_all will divide split_axis into chunks and concatenate
      them along concat_axis. In particular, no dimensions are added or removed.
      False by default.

  Returns:
    When tiled is False, array(s) with shape given by the expression::

      np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)

    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``, i.e. ``axis_size = lax.psum(1, axis_name)``.

    Otherwise array with shape similar to the input shape, except with split_axis
    divided by axis size and concat_axis multiplied by axis size.
  """
  def bind(x, split_axis=split_axis, concat_axis=concat_axis):
    group_size = psum(1, axis_name, axis_index_groups=axis_index_groups)
    if tiled:
      if x.shape[split_axis] % group_size != 0:
        raise ValueError(f"The size of all_to_all split_axis ({x.shape[split_axis]}) "
                         f"has to be divisible by the size of the named axis "
                         f"{axis_name} ({group_size})")
    else:
      if group_size != x.shape[split_axis]:
        msg = ("all_to_all requires the size of the mapped axis axis_name to "
               "equal x.shape[split_axis], but they are {} and {} respectively.")
        raise ValueError(msg.format(group_size, x.shape[split_axis]))
      if split_axis < concat_axis:
        concat_axis += 1  # concat_axis gives a position _after_ split_axis is removed
        x = lax.expand_dims(x, (concat_axis,))  # insert the new axis
      elif split_axis == concat_axis:
        pass
      else:  # concat_axis < split_axis
        x = lax.expand_dims(x, (concat_axis,))  # insert the new axis
        split_axis += 1   # we have a new axis before split_axis now
    result = all_to_all_p.bind(x, split_axis=split_axis, concat_axis=concat_axis,
                               axis_name=axis_name,
                               axis_index_groups=axis_index_groups)
    if not tiled and split_axis != concat_axis:
      result = lax.squeeze(result, (split_axis,))
    return result

  return tree_util.tree_map(bind, x)

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


def pdot(x, y, axis_name, pos_contract=((), ()), pos_batch=((), ()),
         precision=None):
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  return pdot_p.bind(x, y, axis_name=axis_name,
                     pos_contract=pos_contract, pos_batch=pos_batch,
                     precision=lax._canonicalize_precision(precision))


def xeinsum(spec: str, x, y):
  in_spec, out_spec = spec.split('->')
  (lhs_subs, lhs_named), (rhs_subs, rhs_named) = XeinsumSpecParser(in_spec).parse_args()
  (out_subs, out_named), = XeinsumSpecParser(out_spec).parse_args()
  all_named = {*lhs_named, *rhs_named, *out_named}
  all_subs = {*lhs_subs, *rhs_subs, *out_subs}
  lhs_uniques = set(lhs_subs) - set(rhs_subs)
  rhs_uniques = set(rhs_subs) - set(lhs_subs)
  if all_subs & all_named:
    raise NotImplementedError
  if not set(out_named).issubset({*lhs_named, *rhs_named}):
    raise ValueError

  # if a named axis appears in both inputs and not the output, contract!
  named_contract = list(all_named - set(out_named))

  # if a subscript appears in both inputs and not the outputs, contract!
  subs_contract = all_subs - set(out_subs)

  lhs_reduce_axes = [lhs_subs.index(n) for n in lhs_uniques & subs_contract]
  if lhs_reduce_axes:
    x = lax._reduce_sum(x, lhs_reduce_axes)
    for i in sorted(lhs_reduce_axes, reverse=True):
      del lhs_subs[i]

  rhs_reduce_axes = [rhs_subs.index(n) for n in rhs_uniques & subs_contract]
  if rhs_reduce_axes:
    y = lax._reduce_sum(y, rhs_reduce_axes)
    for i in sorted(rhs_reduce_axes, reverse=True):
      del rhs_subs[i]

  pos_contract = unzip2((lhs_subs.index(n), rhs_subs.index(n))
                        for n in subs_contract - (lhs_uniques | rhs_uniques))

  # if a subscript apperas in both inputs _and_ the outputs, batch!
  subs_batch = all_subs - subs_contract
  if subs_batch & (lhs_uniques | rhs_uniques):
    raise NotImplementedError

  pos_batch = unzip2((lhs_subs.index(n), rhs_subs.index(n))
                        for n in subs_batch)

  return pdot(x, y, axis_name=named_contract,
              pos_contract=pos_contract, pos_batch=pos_batch)

class XeinsumSpecParser:
  spec: str
  pos: int

  def __init__(self, spec: str):
    self.spec = spec
    self.pos = 0

  @property
  def eof(self):
    return self.pos == len(self.spec)

  @property
  def cur(self):
    return self.spec[self.pos]

  def parse_subscript(self):
    if self.cur in string.ascii_lowercase:
      out = self.cur
      self.pos += 1
      return out, True
    else:
      return None, False

  def parse_axis_name(self):
    try:
      end = self.spec.index('}', self.pos)
    except ValueError:
      assert False

    try:
      end = self.spec.index(',', self.pos, end)
    except ValueError:
      pass

    axis_name = self.spec[self.pos:end]
    assert axis_name
    self.pos = end
    return axis_name

  def maybe_take(self, char: str, on_eof: bool = False):
    if self.eof:
      return on_eof
    if self.cur == char:
      self.pos += 1
      return True

  def parse_arg(self):
    subscripts = []
    names = []
    while not self.eof:
      subscript, cont = self.parse_subscript()
      if not cont: break
      subscripts.append(subscript)
    if self.eof:
      return False, (subscripts, names)
    if self.maybe_take(','):
      return True, (subscripts, names)
    else:
      assert self.maybe_take('{')
      first = True
      while not self.maybe_take('}'):
        if not first:
          assert self.maybe_take(',')
        first = False
        if self.eof:
          raise ValueError("Unterminated named axis brace")
        axis_name = self.parse_axis_name()
        names.append(axis_name)
      return self.maybe_take(',', False), (subscripts, names)

  def parse_args(self):
    arg_specs = []
    cont = True
    while not self.eof:
      cont, result = self.parse_arg()
      arg_specs.append(result)
    if cont:
      arg_specs.append(([], []))
    return arg_specs


def pgather(src, idx, axes: Union[int, AxisName]):
  """Uses the last positional axis of idx to index into src's axes."""
  if not isinstance(axes, (tuple, list)):
    axes = (axes,)
  # TODO: Canonicalize exes!
  return pgather_p.bind(src, idx, axes=tuple(axes))


### parallel primitives

def _subst_all_names_in_param(
    pname: str, params: core.ParamDict, subst: core.AxisSubst, traverse: bool) -> core.ParamDict:
  axis_name = params[pname]
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  result = dict(params)
  result[pname] = sum(((name,) if isinstance(name, int) else subst(name)
                       for name in axis_name),
                      ())
  return result

def _reduction_with_positional_batcher(prim, vals_in, dims_in, axis_index_groups,
    transform_unmapped, transform_mapped):
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not supported in vmap collectives. "
                              "Please open a feature request!")
  vals_in = [val if d is batching.not_mapped or d == 0 else _moveaxis(d, 0, val)
             for val, d in zip(vals_in, dims_in)]
  mapped_vals_in, unmapped_vals_in = partitioned_vals_in = [], []
  mapped_idxs, unmapped_idxs = partitioned_idxs = [], []
  for i, (val, d) in enumerate(zip(vals_in, dims_in)):
    partitioned_vals_in[d is batching.not_mapped].append(val)
    partitioned_idxs[d is batching.not_mapped].append(i)
  vals_out = [None] * len(vals_in)
  if unmapped_vals_in:
    unmapped_axes, unmapped_vals_in = transform_unmapped(0, unmapped_vals_in)
    unmapped_vals_out = prim.bind(*unmapped_vals_in, axes=unmapped_axes, axis_index_groups=None)
    for i, val in zip(unmapped_idxs, unmapped_vals_out):
      vals_out[i] = val
  if mapped_vals_in:
    mapped_axes, mapped_vals_in = transform_mapped(0, mapped_vals_in)
    mapped_vals_out = prim.bind(*mapped_vals_in, axes=mapped_axes, axis_index_groups=None)
    for i, val in zip(mapped_idxs, mapped_vals_out):
      vals_out[i] = val
  assert all(v is not None for v in vals_out)
  return vals_out

def _reduction_batcher(prim, vals_in, dims_in, *, axes, axis_index_groups):
  assert prim.multiple_results
  if not any(isinstance(axis, int) for axis in axes):
    return prim.bind(*vals_in, axes=axes, axis_index_groups=axis_index_groups), dims_in
  vals_out = _reduction_with_positional_batcher(
      prim, vals_in, dims_in, axis_index_groups,
      lambda d, d_vals_in: (axes, d_vals_in),
      lambda d, d_vals_in: (tuple(axis + (axis >= d) if isinstance(axis, int) else axis
                                  for axis in axes),
                            d_vals_in))
  # _reduction_with_positional_batcher moves all map dims to 0
  return vals_out, [d if d is batching.not_mapped else 0 for d in dims_in]

def _batched_reduction_collective(
    prim, if_unmapped, frame, vals_in, dims_in, axes,
    axis_index_groups):
  assert prim.multiple_results
  assert frame.name in axes
  # Note that we have a choice here. We can either unfuse the reduction into one
  # that handles the batched dims and then another one that handles the rest.
  # Alternatively, we can keep the dimension reduction fused with the rest, but
  # we have to split the primitive into one for unmapped inputs and another
  # one for mapped, because they differ in their `axes` parameter.
  # We choose the second strategy here.
  vals_out = _reduction_with_positional_batcher(
      prim, vals_in, dims_in, axis_index_groups,
      lambda d, d_vals_in: (tuple(axis for axis in axes if axis != frame.name),
                            [if_unmapped(v, frame.size) for v in d_vals_in]),
      lambda d, d_vals_in: (tuple(axis + (axis >= d) if isinstance(axis, int) else
                                  axis if axis != frame.name else
                                  d
                                  for axis in axes),
                            d_vals_in))
  return vals_out, [batching.not_mapped] * len(vals_out)

def _replica_groups(axis_env, axis_name, axis_index_groups):
  replica_groups = xla.axis_groups(axis_env, axis_name)
  if axis_index_groups is not None:
    replica_groups = [[axis_group[i] for i in axis_index_group]
                      for axis_group in replica_groups
                      for axis_index_group in axis_index_groups]
  return replica_groups

def _allreduce_impl(pos_reducer, *args, axes, axis_index_groups):
  assert axis_index_groups is None
  assert all(isinstance(axis, int) for axis in axes)
  return [pos_reducer(arg, axes) for arg in args]

def _allreduce_abstract_eval(*args, axes, axis_index_groups):
  # TODO(frostig,mattjj,jekbradbury): maybe check aval names here
  pos_axes = tuple(axis for axis in axes if isinstance(axis, int))
  named_shapes = [arg.named_shape for arg in args]
  if axis_index_groups is None:
    named_axes = set(axis for axis in axes if not isinstance(axis, int))
    named_shapes = [{name: size for name, size in arg.named_shape.items()
                     if name not in named_axes} for arg in args]
  else:
    if len(pos_axes) != 0:
      raise ValueError(f"axis_index_groups can only be used with reductions over "
                       f"named axes, but got: {axes}")
  return [ShapedArray(lax._reduce_op_shape_rule(raise_to_shaped(arg), axes=pos_axes),
                      arg.dtype, named_shape=named_shape)
          for arg, named_shape in zip(args, named_shapes)]

def _allreduce_translation_rule(prim, pos_prim, c, *args, axes, axis_index_groups,
                                axis_env, platform):
  if axis_index_groups is not None and platform == "tpu":
    len_0 = len(axis_index_groups[0])
    if any(len(g) != len_0 for g in axis_index_groups):
      raise ValueError("axis_index_groups must all be the same size")
  named_axes, positional_axes = axes_partition = [], []
  for axis in axes:
    axes_partition[isinstance(axis, int)].append(axis)

  if positional_axes:
    args = map(partial(xla.translations[pos_prim], c, axes=tuple(positional_axes)), args)
  if not named_axes:
    return xops.Tuple(c, args)

  def all_reduce(x):
    replica_groups_protos = xc.make_replica_groups(
        _replica_groups(axis_env, named_axes, axis_index_groups))
    scalar = ShapedArray((), c.get_shape(x).numpy_dtype())
    computation = xla.primitive_subcomputation(prim, scalar, scalar)
    return xops.AllReduce(x, computation, replica_groups_protos, None, None)

  if prim is not lax.add_p:
    outs = [all_reduce(x) for x in args]
  else:
    # TODO(b/141575627): we handle complex-dtype sum-reduction directly as a
    # special case because it's not currently handled by XLA:GPU
    outs = [xops.Complex(all_reduce(xops.Real(x)), all_reduce(xops.Imag(x)))
            if dtypes.issubdtype(c.get_shape(x).numpy_dtype(), np.complexfloating)
            else all_reduce(x) for x in args]
  return xops.Tuple(c, outs)

def _psum_transpose_rule(cts, *args, axes, axis_index_groups):
  named_axes, pos_axes = axes_partition = [], []
  for axis in axes:
    axes_partition[isinstance(axis, int)].append(axis)

  if pos_axes:
    def broadcast_positional(ct, arg):
      assert ad.is_undefined_primal(arg)
      if type(ct) is ad.Zero: return ad.Zero(arg.aval)
      return lax._reduce_sum_transpose_rule(ct, arg, axes=pos_axes)[0]
    cts = map(broadcast_positional, cts, args)

  # We treat psum as psum + pbroadcast, which is why the transpose reduces
  # over the named axes again (unlike for positional axes).
  nonzero_out_cts, treedef = tree_util.tree_flatten(cts)
  nonzero_in_cts = psum_p.bind(*nonzero_out_cts, axes=named_axes,
                               axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, nonzero_in_cts)

psum_p = core.AxisPrimitive('psum')
psum_p.multiple_results = True
psum_p.def_impl(partial(_allreduce_impl, lax._reduce_sum))
psum_p.def_abstract_eval(_allreduce_abstract_eval)
xla.parallel_translations[psum_p] = partial(_allreduce_translation_rule,
                                            lax.add_p, lax.reduce_sum_p)  # type: ignore
ad.deflinear2(psum_p, _psum_transpose_rule)
pxla.multi_host_supported_collectives.add(psum_p)
batching.primitive_batchers[psum_p] = partial(_reduction_batcher, psum_p)
batching.collective_rules[psum_p] = \
  partial(_batched_reduction_collective, psum_p, lambda v, axis_size: axis_size * v)
core.axis_substitution_rules[psum_p] = partial(_subst_all_names_in_param, 'axes')

# We set a special bind rule for psum so that psum(1, 'i') can be evaluated at
# tracing time.
@psum_p.def_custom_bind
def psum_bind(*args, axes, axis_index_groups):
  if all(not isinstance(x, core.Tracer) for x in args):
    named_axes, pos_axes = axes_partition = [], []
    for axis in axes:
      axes_partition[isinstance(axis, int)].append(axis)
    def pos_reduce(x):
      if not pos_axes:
        return x
      return lax._reduce_sum(x, [canonicalize_axis(axis, getattr(x, 'ndim', 0))
                                 for axis in pos_axes])
    if axis_index_groups is not None:
      assert not pos_axes
      size = len(axis_index_groups[0])
    else:
      size = prod([core.axis_frame(name).size for name in named_axes])  # type: ignore
    return tuple(lax._const(x, size) * pos_reduce(x) for x in args)
  return core.AxisPrimitive.bind(
      psum_p, *args, axes=axes, axis_index_groups=axis_index_groups)


pmax_p = core.AxisPrimitive('pmax')
pmax_p.multiple_results = True
pmax_p.def_impl(partial(_allreduce_impl, lax._reduce_max))
pmax_p.def_abstract_eval(_allreduce_abstract_eval)
xla.parallel_translations[pmax_p] = partial(_allreduce_translation_rule,
                                            lax.max_p, lax.reduce_max_p)  # type: ignore
pxla.multi_host_supported_collectives.add(pmax_p)
batching.primitive_batchers[pmax_p] = partial(_reduction_batcher, pmax_p)
batching.collective_rules[pmax_p] = \
  partial(_batched_reduction_collective, pmax_p, lambda v, axis_size: v)
core.axis_substitution_rules[pmax_p] = partial(_subst_all_names_in_param, 'axes')


pmin_p = core.AxisPrimitive('pmin')
pmin_p.multiple_results = True
pmin_p.def_impl(partial(_allreduce_impl, lax._reduce_min))
pmin_p.def_abstract_eval(_allreduce_abstract_eval)
xla.parallel_translations[pmin_p] = partial(_allreduce_translation_rule,
                                            lax.min_p, lax.reduce_min_p)  # type: ignore
pxla.multi_host_supported_collectives.add(pmin_p)
batching.primitive_batchers[pmin_p] = partial(_reduction_batcher, pmin_p)
batching.collective_rules[pmin_p] = \
  partial(_batched_reduction_collective, pmin_p, lambda v, axis_size: v)
core.axis_substitution_rules[pmin_p] = partial(_subst_all_names_in_param, 'axes')


def _ppermute_translation_rule(c, x, *, axis_name, axis_env, perm, platform):
  replica_groups = _replica_groups(axis_env, axis_name, None)
  group_size = len(replica_groups[0])
  srcs, dsts = unzip2((src % group_size, dst % group_size) for src, dst in perm)
  if not (len(srcs) == len(set(srcs)) and len(dsts) == len(set(dsts))):
    msg = "ppermute sources and destinations must be unique, got {}."
    raise ValueError(msg.format(perm))

  full_perm = []
  for grp in replica_groups:
    grp = list(sorted(grp))
    full_perm.extend((grp[src], grp[dst]) for src, dst in perm)
  return xops.CollectivePermute(x, full_perm)

def _ppermute_transpose_rule(t, x, perm, axis_name):
  srcs, dsts = unzip2(perm)
  inverse_perm = list(zip(dsts, srcs))
  return [ppermute(t, axis_name=axis_name, perm=inverse_perm)]

def _ppermute_batcher(frame, vals_in, dims_in, axis_name, perm):
  (v,), (d,) = vals_in, dims_in
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  remaining_axes = tuple(axis for axis in axis_name if axis != frame.name)
  if frame.size == 1 and remaining_axes:
    return ppermute_p.bind(v, perm=perm, axis_name=remaining_axes), d
  if remaining_axes:
    raise NotImplementedError("ppermute batcher only supports a single axis")
  assert axis_name[0] == frame.name, "ppermute batcher called with a wrong axis!"
  assert len(perm) == frame.size, "Permutation doesn't match the axis size!"
  assert d is not batching.not_mapped
  perm_indices = [None] * frame.size
  for src, dst in perm:
    perm_indices[src] = dst
  return lax_numpy.take(v, perm_indices, d), d

def _collective_batcher(prim, args, dims, **params):
  return prim.bind(*args, **params), dims if prim.multiple_results else dims[0]

ppermute_p = core.AxisPrimitive('ppermute')
ppermute_p.def_abstract_eval(lambda x, **params: raise_to_shaped(x))
ad.deflinear2(ppermute_p, _ppermute_transpose_rule)
xla.parallel_translations[ppermute_p] = _ppermute_translation_rule
pxla.multi_host_supported_collectives.add(ppermute_p)
batching.primitive_batchers[ppermute_p] = partial(_collective_batcher, ppermute_p)
batching.collective_rules[ppermute_p] = _ppermute_batcher
core.axis_substitution_rules[ppermute_p] = partial(_subst_all_names_in_param, 'axis_name')


def _moveaxis(src, dst, x):
  perm = [i for i in range(x.ndim) if i != src]
  perm.insert(dst, src)
  return lax.transpose(x, perm)

def _splitaxis(axis, factor, x):
  new_shape = list(x.shape)
  assert new_shape[axis] % factor == 0, (new_shape[axis], factor)
  new_shape[axis:axis+1] = [factor, new_shape[axis] // factor]
  return x.reshape(new_shape)

def _foldaxis(axis, x):
  new_shape = list(x.shape)
  new_shape[axis:axis+2] = [x.shape[axis] * x.shape[axis + 1]]
  return x.reshape(new_shape)

def _index_in_group(axis_name, axis_index_groups):
  cur_device_id = axis_index(axis_name)
  if axis_index_groups is None:
    return cur_device_id
  # We use argsort to invert the axis_index_groups permutation
  flat_groups = np.array(axis_index_groups).flatten()
  device_id_to_idx = flat_groups.argsort() % len(axis_index_groups[0])
  return lax.squeeze(lax.dynamic_slice_in_dim(device_id_to_idx, cur_device_id, 1), [0])

def _all_to_all_via_all_gather(x, *, axis_name, split_axis, concat_axis, axis_index_groups):
  idx = _index_in_group(axis_name, axis_index_groups)
  full = all_gather(x, axis_name, axis_index_groups=axis_index_groups)
  axis_size = full.shape[0]
  tile_size = x.shape[split_axis] // axis_size
  tile_base_idx = idx * tile_size
  sliced = lax.dynamic_slice_in_dim(full, tile_base_idx, tile_size, split_axis + 1)
  return _foldaxis(concat_axis, _moveaxis(0, concat_axis, sliced))

def _all_to_all_translation_rule(c, x, *, split_axis, concat_axis, axis_name,
                                 axis_index_groups, axis_env, platform):
  # Workaround for AllToAll not being implemented on CPU.
  replica_groups = _replica_groups(axis_env, axis_name, axis_index_groups)
  if len(replica_groups[0]) == 1:
    return x
  elif (platform == "tpu") or ((platform == "gpu") and (split_axis == 0) and
                               (concat_axis == 0)):
    split_count = len(replica_groups[0])
    if not all(split_count == len(g) for g in replica_groups):
      raise ValueError('Replica groups must be equally sized')
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    return xops.AllToAll(x, split_axis, concat_axis, split_count, replica_groups_protos)
  else:
    warnings.warn(
        "all_to_all (and pswapaxes) are only implemented properly for TPUs and GPUs (if "
        "split_axis and concat_axis are both 0). All other backends emulate it using a "
        "very slow and memory intensive algorithm, so expect significant slowdowns."
    )
    lowering = xla.lower_fun(
        _all_to_all_via_all_gather, multiple_results=False, parallel=True)
    return lowering(
        c,
        x,
        split_axis=split_axis,
        concat_axis=concat_axis,
        axis_name=axis_name,
        axis_index_groups=axis_index_groups,
        axis_env=axis_env,
        platform=platform)

def _all_to_all_transpose_rule(cts, x, axis_name, split_axis, concat_axis, axis_index_groups):
  return (all_to_all(
      cts,
      axis_name=axis_name,
      split_axis=concat_axis,
      concat_axis=split_axis,
      axis_index_groups=axis_index_groups),)

def _all_to_all_batcher(vals_in, dims_in, *, axis_name, split_axis, concat_axis, axis_index_groups):
  x, = vals_in
  d, = dims_in
  result = all_to_all_p.bind(
      x,
      axis_name=axis_name,
      split_axis=split_axis + (d <= split_axis),
      concat_axis=concat_axis + (d <= concat_axis),
      axis_index_groups=axis_index_groups)
  return result, d

def _all_to_all_batched_collective(frame, vals_in, dims_in,
                                   axis_name, split_axis, concat_axis,
                                   axis_index_groups):
  if axis_index_groups is not None:
    raise NotImplementedError("Please open a feature request!")
  x, = vals_in
  d, = dims_in
  if isinstance(axis_name, (list, tuple)):
    pos = axis_name.index(frame.name)
    major_axes, minor_axes = axis_name[:pos], axis_name[pos + 1:]
  else:
    major_axes, minor_axes = (), ()
  # Optimized case when no splitting is necessary
  if not major_axes and not minor_axes:
    if split_axis == concat_axis:
      axis = split_axis + (d <= split_axis)
      d_pre_split = d
      x = _splitaxis(axis, frame.size, x)
      d += (axis <= d)
      return _foldaxis(axis, moveaxis(x, (d, axis), (axis, d))), d_pre_split
    else:
      x_concat = _foldaxis(concat_axis, _moveaxis(d, concat_axis, x))
      return _splitaxis(split_axis, frame.size, x_concat), split_axis
  # Here we have to handle either the major or the minor dimensions
  # We will be accumulating chunks into the three leading dims: [Major, Current, Minor, ...]
  x, d = lax.expand_dims(_moveaxis(d, 0, x), (0, 2)), 1
  split_axis += 3; concat_axis += 3  # Offset by extra three leading dims

  if major_axes:
    x = all_to_all_p.bind(x, axis_name=major_axes,
                          split_axis=split_axis, concat_axis=0,
                          axis_index_groups=axis_index_groups)
  # Split out the local part into axis new_d (NOTE: d is already in axis 1)
  x = _splitaxis(split_axis, frame.size, x)
  new_d = split_axis
  concat_axis += (split_axis <= concat_axis)  # Offset the existing axes by the new batch axis
  split_axis += 1
  if minor_axes:
    x = all_to_all_p.bind(x, axis_name=minor_axes,
                          split_axis=split_axis, concat_axis=2,
                          axis_index_groups=axis_index_groups)

  # Fold the chunk axes into a single one
  x = _foldaxis(0, _foldaxis(0, x))
  split_axis -= 2; concat_axis -= 2; new_d -= 2
  # Fold gathered axes into concat_axis
  x = _foldaxis(concat_axis - 1, _moveaxis(0, concat_axis - 1, x))
  new_d -= 1  # We've removed 0th dimension, so new_d needs to be adjusted
  return x, new_d

def _all_to_all_abstract_eval(x, axis_name, split_axis, concat_axis, axis_index_groups):
  input_aval = raise_to_shaped(x)
  shape = list(input_aval.shape)
  axis_size = psum(1, axis_name) if axis_index_groups is None else len(axis_index_groups[0])
  assert shape[split_axis] % axis_size == 0, (shape[split_axis], axis_size)
  shape[split_axis] //= axis_size
  shape[concat_axis] *= axis_size
  return input_aval.update(shape=tuple(shape), weak_type=False)

all_to_all_p = core.AxisPrimitive('all_to_all')
all_to_all_p.def_abstract_eval(_all_to_all_abstract_eval)
xla.parallel_translations[all_to_all_p] = _all_to_all_translation_rule
ad.deflinear2(all_to_all_p, _all_to_all_transpose_rule)
pxla.multi_host_supported_collectives.add(all_to_all_p)
batching.primitive_batchers[all_to_all_p] = _all_to_all_batcher
batching.collective_rules[all_to_all_p] = _all_to_all_batched_collective
core.axis_substitution_rules[all_to_all_p] = partial(_subst_all_names_in_param, 'axis_name')


def all_gather(x, axis_name, *, axis_index_groups=None, axis=0, tiled=False):
  """Gather values of x across all replicas.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This is equivalent to, but faster than, all_to_all(broadcast(x)).

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run all gather over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.
    axis: a positional axis into which the chunks along ``axis_name`` will be
      concatenated.
    tiled: when ``False``, the chunks will be stacked into a fresh positional
      axis at index ``axis`` in the output. When ``True``, ``axis`` has to
      refer to an existing positional dimension and the chunks will be
      concatenated into that dimension.

  Returns:
    Array(s) representing the result of an all-gather along the axis
    ``axis_name``. Shapes are the same as ``x.shape``, but:

    - when ``tiled`` is ``False``, there is a new dimension equal to the
      size of axis ``axis_name`` in position ``axis``,
    - when ``tiled`` is ``True``, the size of dimension in position ``axis``
      is multiplied by the size of axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.all_gather(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [[0 1 2 3]
   [0 1 2 3]
   [0 1 2 3]
   [0 1 2 3]]

  An example of using axis_index_groups, groups split by even & odd device ids:

  >>> x = np.arange(16).reshape(4, 4)
  >>> print(x)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
  >>> def f(x):
  ...   return jax.lax.all_gather(
  ...       x, 'i', axis_index_groups=[[0, 2], [3, 1]])
  >>> y = jax.pmap(f, axis_name='i')(x)
  >>> print(y)
  [[[ 0  1  2  3]
    [ 8  9 10 11]]
   [[12 13 14 15]
    [ 4  5  6  7]]
   [[ 0  1  2  3]
    [ 8  9 10 11]]
   [[12 13 14 15]
    [ 4  5  6  7]]]
  """
  axis_size = psum(1, axis_name, axis_index_groups=axis_index_groups)
  bind = partial(all_gather_p.bind, all_gather_dimension=axis,
                 axis_name=axis_name, axis_index_groups=axis_index_groups,
                 axis_size=axis_size, tiled=tiled)
  return tree_util.tree_map(bind, x)

def _expand(dim, size, index, tiled, x):
  shape = list(x.shape)
  if tiled:
    tile_size = shape[dim]
    shape[dim] *= size
    out = lax.full(shape, lax._const(x, 0))
    return lax.dynamic_update_slice_in_dim(out, x, index * tile_size, dim)
  else:
    shape.insert(dim, size)
    out = lax.full(shape, lax._const(x, 0))
    return lax.dynamic_update_index_in_dim(out, x, index, dim)

def _all_gather_via_psum(x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  index = _index_in_group(axis_name, axis_index_groups)
  outs = tree_util.tree_map(partial(_expand, all_gather_dimension, axis_size, index, tiled), x)
  return psum(outs, axis_name, axis_index_groups=axis_index_groups)

def _all_gather_impl(x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  raise AssertionError("Unexpected call to _all_gather_impl")

def _all_gather_translation_rule(c, x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled, axis_env, platform):
  # TODO(cjfj): Enable this for TPU also?
  if (platform == 'gpu') and (all_gather_dimension == 0):
    if not tiled:
      new_shape = list(c.get_shape(x).dimensions())
      new_shape.insert(all_gather_dimension, 1)
      broadcast_dimensions = [i for i in range(len(new_shape)) if i != all_gather_dimension]
      x = xops.BroadcastInDim(x, new_shape, broadcast_dimensions)
    replica_groups = _replica_groups(axis_env, axis_name, axis_index_groups)
    return xops.AllGather(x, all_gather_dimension=all_gather_dimension, shard_count=axis_size,
                          replica_groups=xc.make_replica_groups(replica_groups))
  else:
    lowering = xla.lower_fun(_all_gather_via_psum, multiple_results=False, parallel=True)
    return lowering(c, x, all_gather_dimension=all_gather_dimension, axis_name=axis_name,
                    axis_index_groups=axis_index_groups, axis_size=axis_size, tiled=tiled,
                    axis_env=axis_env, platform=platform)

def _all_gather_abstract_eval(x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  x_aval = raise_to_shaped(x)
  new_shape = list(x_aval.shape)
  if tiled:
    new_shape[all_gather_dimension] *= axis_size
  else:
    new_shape.insert(all_gather_dimension, axis_size)
  new_named_shape = {name: size for name, size in x_aval.named_shape.items()
                     if name not in axis_name}
  return x_aval.update(shape=new_shape, named_shape=new_named_shape)

def _all_gather_transpose_rule(cts, x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  if tiled:
    raise NotImplementedError("Please open a feature request!")
  # TODO(cjfj): Add reduce-scatter op to XLA?
  concat_axis = 0
  return (lax_numpy.sum(all_to_all(
      cts, axis_name=axis_name, split_axis=all_gather_dimension,
      concat_axis=concat_axis, axis_index_groups=axis_index_groups),
      axis=concat_axis),)

def _all_gather_batcher(vals_in, dims_in, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  if tiled:
    raise NotImplementedError("Please open a feature request!")
  (x,), (d,) = vals_in, dims_in
  if d <= all_gather_dimension:
    all_gather_dimension += 1
  else:
    d += 1
  result = all_gather_p.bind(
      x,
      all_gather_dimension=all_gather_dimension,
      axis_name=axis_name,
      axis_index_groups=axis_index_groups,
      axis_size=axis_size,
      tiled=tiled)
  return result, d

def _all_gather_batched_collective(frame, vals_in, dims_in, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  if tiled:
    raise NotImplementedError("Please open a feature request!")
  assert axis_index_groups is None, "axis_index_groups not supported in vmap"
  assert axis_size == frame.size, "axis size doesn't match"
  if not isinstance(axis_name, tuple):
    axis_name = (axis_name,)
  if len(axis_name) > 1:
    raise NotImplementedError("Please open a feature request!")
  assert axis_name == (frame.name,), "batcher called with wrong axis name"
  (x,), (d,) = vals_in, dims_in
  if d is batching.not_mapped:
    out_shape = list(np.shape(x))
    out_shape.insert(all_gather_dimension, axis_size)
    broadcast_dims = [i for i in range(len(out_shape)) if i != all_gather_dimension]
    return lax.broadcast_in_dim(x, out_shape, broadcast_dims), batching.not_mapped
  return _moveaxis(d, all_gather_dimension, x), batching.not_mapped

all_gather_p = core.AxisPrimitive('all_gather')
all_gather_p.def_abstract_eval(_all_gather_abstract_eval)
all_gather_p.def_impl(_all_gather_impl)
xla.parallel_translations[all_gather_p] = _all_gather_translation_rule
ad.deflinear2(all_gather_p, _all_gather_transpose_rule)
pxla.multi_host_supported_collectives.add(all_gather_p)
batching.primitive_batchers[all_gather_p] = _all_gather_batcher
batching.collective_rules[all_gather_p] = _all_gather_batched_collective
core.axis_substitution_rules[all_gather_p] = partial(_subst_all_names_in_param, 'axis_name')


def _reduce_scatter_via_reducer(x, *, reducer, scatter_dimension, axis_name, axis_index_groups, axis_size, tiled):
  index = _index_in_group(axis_name, axis_index_groups)
  scatter_dim_input_size = x.shape[scatter_dimension]
  if tiled and scatter_dim_input_size % axis_size != 0:
      raise ValueError(f"tiled reduce_scatter operand scatter dimension size "
                       f"{scatter_dim_input_size} must be divisible by "
                       f"shard count {axis_size}")
  elif not tiled and scatter_dim_input_size != axis_size:
      raise ValueError(f"reduce_scatter operand scatter dimension size "
                       f"{scatter_dim_input_size} must match shard count"
                       f"{axis_size}")
  scatter_dim_output_size = scatter_dim_input_size // axis_size

  outs = reducer(x, axis_name=axis_name, axis_index_groups=axis_index_groups)
  outs = lax.dynamic_slice_in_dim(
      outs,
      start_index=index * scatter_dim_output_size,
      slice_size=scatter_dim_output_size,
      axis=scatter_dimension)
  if not tiled:
    outs = lax.squeeze(outs, [scatter_dimension])
  return outs


def _reduce_scatter_translation_rule(prim, reducer, c, x, *, scatter_dimension, axis_name,axis_index_groups, axis_size, tiled, axis_env, platform):
  # TODO(b/194706412): Enable this for TPU?
  if platform == "gpu":
    scalar = ShapedArray((), c.get_shape(x).numpy_dtype())
    computation = xla.primitive_subcomputation(prim, scalar, scalar)
    replica_groups = _replica_groups(axis_env, axis_name, axis_index_groups)
    x = xops.ReduceScatter(
        x,
        computation,
        scatter_dimension=scatter_dimension,
        shard_count=axis_size,
        replica_groups=xc.make_replica_groups(replica_groups))
    if not tiled:
      new_shape = list(c.get_shape(x).dimensions())
      del new_shape[scatter_dimension]
      x = xops.Reshape(x, new_shape)
    return x
  else:
    return xla.lower_fun(
        _reduce_scatter_via_reducer, multiple_results=False, parallel=True)(
            c,
            x,
            reducer=reducer,
            scatter_dimension=scatter_dimension,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups,
            axis_size=axis_size,
            tiled=tiled,
            axis_env=axis_env,
            platform=platform)


def _reduce_scatter_abstract_eval(x, *, axis_name, scatter_dimension,
                                  axis_index_groups, axis_size, tiled):
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  x_aval = core.raise_to_shaped(x)
  new_shape = list(x_aval.shape)
  scatter_dim_input_size = x_aval.shape[scatter_dimension]
  if tiled:
    if scatter_dim_input_size % axis_size != 0:
      raise ValueError(f"tiled reduce_scatter operand scatter dimension size "
                       f"{scatter_dim_input_size} must be divisible by "
                       f"shard_count {axis_size}")
    new_shape[scatter_dimension] = scatter_dim_input_size // axis_size
  else:
    if scatter_dim_input_size != axis_size:
      raise ValueError(f"reduce_scatter operand scatter dimension size "
                       f"{scatter_dim_input_size} must match shard count "
                       f"{axis_size}")
    del new_shape[scatter_dimension]

  new_named_shape = {
      name: size
      for name, size in x_aval.named_shape.items()
      if name not in axis_name
  }
  return x_aval.update(shape=new_shape, named_shape=new_named_shape)


reduce_scatter_p = core.AxisPrimitive("reduce_scatter")
reduce_scatter_p.def_abstract_eval(_reduce_scatter_abstract_eval)
xla.parallel_translations[reduce_scatter_p] = partial(
    _reduce_scatter_translation_rule, lax.add_p, psum)  # type: ignore
pxla.multi_host_supported_collectives.add(reduce_scatter_p)


def psum_scatter(x, axis_name, *, scatter_dimension=0, axis_index_groups=None, tiled=False):
  """Compute an all-reduce sum over the axis ``axis_name``, and scatter the result.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    scatter_dimension: a positional axis into which the all reduce result along
      ``axis_name`` will be scattered.
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run reduce-scatter over the
      first two and the last two replicas). Groups must cover all axis indices
      exactly once, and all groups must be the same size.
    tiled: when ``False``, the size of dimension in ``scatter_dimension`` must
      match the size of axis ``axis_name`` (or the group size if
      ``axis_index_groups`` is given). After scattering the all reduce result
      along ``scatter_dimension``, the output is sequeezed by removing
      ``scatter_dimension``. When ``True``, the size of dimension in
      ``scatter_dimension` must be dividible by the size of axis ``axis_name``
      (or the group size if ``axis_index_groups`` is given),
      and ``scatter_dimension`` is preserved.

  Returns:
    Array(s) with the similar shape as ``x``, except the size of dimension in
    position``scatter_dimension`` is divided by the size of axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(16).reshape(4,4)
  >>> print(x)
  [[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]
   [12 13 14 15]]
  >>> y = jax.pmap(lambda x: jax.lax.psum_scatter(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [24 28 32 36]

  if using tiled:

  >>> y = jax.pmap(lambda x: jax.lax.psum_scatter(x, 'i', tiled=True), axis_name='i')(x)
  >>> print(y)
  [[24]
   [28]
   [32]
   [36]]

  An example of using axis_index_groups:

  >>> def f(x):
  ...   return jax.lax.psum_scatter(
  ...       x, 'i', axis_index_groups=[[0, 2], [3, 1]], tiled=True)
  >>> y = jax.pmap(f, axis_name='i')(x)
  >>> print(y)
  [[ 8 10]
   [20 22]
   [12 14]
   [16 18]]
  """
  axis_size = psum(1, axis_name, axis_index_groups=axis_index_groups)
  bind = partial(
      reduce_scatter_p.bind,
      axis_name=axis_name,
      scatter_dimension=scatter_dimension,
      axis_index_groups=axis_index_groups,
      axis_size=axis_size,
      tiled=tiled)
  return tree_util.tree_map(bind, x)


def _axis_index_translation_rule(c, *, axis_name, axis_env, platform):
  axis_pos = list(axis_env.names).index(axis_name)
  nreplicas = axis_env.nreps // prod(axis_env.sizes)
  div = xb.constant(c, np.array(nreplicas * prod(axis_env.sizes[axis_pos+1:]),
                                dtype=np.uint32))
  mod = xb.constant(c, np.array(axis_env.sizes[axis_pos], dtype=np.uint32))
  unsigned_index = xops.Rem(xops.Div(xops.ReplicaId(c), div), mod)
  return xops.ConvertElementType(unsigned_index, xb.dtype_to_etype(np.int32))

def _axis_index_abstract_eval(*, axis_name):
  frame = core.axis_frame(axis_name)
  return ShapedArray((), np.int32, named_shape={axis_name: frame.size})

axis_index_p = core.Primitive('axis_index')
xla.parallel_translations[axis_index_p] = _axis_index_translation_rule
axis_index_p.def_abstract_eval(_axis_index_abstract_eval)
pxla.multi_host_supported_collectives.add(axis_index_p)
core.axis_substitution_rules[axis_index_p] = partial(_subst_all_names_in_param, 'axis_name')

# Axis index doesn't get any arguments, so that the default bind would have no
# way to call into a data-dependency based trace such as vmap. Each trace that
# wants to bind an axis name has to additionally implement `process_axis_index`
# and put its main trace on the axis env stack.
def _axis_index_bind(*, axis_name):
  def name_idx(name):
    frame = core.axis_frame(name)
    dynamic = core.thread_local_state.trace_state.trace_stack.dynamic
    if (frame.main_trace is None or dynamic.level > frame.main_trace.level):
      return core.Primitive.bind(axis_index_p, axis_name=name)
    else:
      trace = frame.main_trace.with_cur_sublevel()
      return trace.process_axis_index(frame)

  if not isinstance(axis_name, (tuple, list)):
    return name_idx(axis_name)
  else:
    inner_size = 1
    index = 0
    for name in reversed(axis_name):
      index += name_idx(name) * inner_size
      inner_size *= psum(1, name)
    return index
axis_index_p.def_custom_bind(_axis_index_bind)

def _vmap_process_axis_index(self, frame):
  assert frame.size is not None
  return batching.BatchTracer(self, lax.iota(np.int32, frame.size), 0)
batching.BatchTrace.process_axis_index = _vmap_process_axis_index  # type: ignore


pdot_p = core.AxisPrimitive('pdot')
core.axis_substitution_rules[pdot_p] = partial(_subst_all_names_in_param, 'axis_name')

@pdot_p.def_impl
def _pdot_impl(x, y, *, axis_name, pos_contract, pos_batch, precision):
  if axis_name: raise NameError(f"unbound axis name: {axis_name[0]}")
  return lax.dot_general(x, y, [pos_contract, pos_batch], precision=precision)

@pdot_p.def_abstract_eval
def _pdot_abstract_eval(x, y, *, axis_name, pos_contract, pos_batch, precision):
  # TODO(frostig,mattjj,jekbradbury): check inputs have given axis names?
  if not len(set(axis_name)) == len(axis_name): raise ValueError
  pos_aval = lax.dot_general_p.abstract_eval(
      x, y, dimension_numbers=[pos_contract, pos_batch],
      precision=None, preferred_element_type=None)
  common_named_shape = core.join_named_shapes(x.named_shape, y.named_shape)
  named_shape = {name: size
                 for name, size in common_named_shape.items()
                 if name not in axis_name}
  return pos_aval.update(named_shape=named_shape)

def _pdot_vmap_collective_rule(frame, vals_in, dims_in, *, axis_name,
                               pos_contract, pos_batch, precision):
  x, y = vals_in
  x_dim, y_dim = dims_in
  x_pos_contract, y_pos_contract = pos_contract
  x_pos_contract = [x_dim] + [d + (d >= x_dim) for d in x_pos_contract]
  y_pos_contract = [y_dim] + [d + (d >= y_dim) for d in y_pos_contract]
  x_pos_batch, y_pos_batch = pos_batch
  x_pos_batch = [d + (d >= x_dim) for d in x_pos_batch]
  y_pos_batch = [d + (d >= y_dim) for d in y_pos_batch]
  remaining_axis_names = tuple(n for n in axis_name if n != frame.name)
  out = pdot_p.bind(x, y, axis_name=remaining_axis_names,
                    pos_contract=[x_pos_contract, y_pos_contract],
                    pos_batch=[x_pos_batch, y_pos_batch], precision=precision)
  return out, None
batching.collective_rules[pdot_p] = _pdot_vmap_collective_rule

def _pdot_vmap_batching_rule(vals_in, dims_in, *, axis_name, pos_contract,
                             pos_batch, precision):
  x, y = vals_in
  (pos_contract, pos_batch), result_batch_dim = lax._dot_general_batch_dim_nums(
      (x.ndim, y.ndim), dims_in, [pos_contract, pos_batch])
  out = pdot_p.bind(x, y, axis_name=axis_name, pos_contract=pos_contract,
                    pos_batch=pos_batch, precision=precision)
  return out, result_batch_dim
batching.primitive_batchers[pdot_p] = _pdot_vmap_batching_rule

def _pdot_translation_rule(c, x, y, *, axis_name, pos_contract, pos_batch,
                           precision, axis_env, platform):
  local_out = lax._dot_general_translation_rule(
      c, x, y, dimension_numbers=[pos_contract, pos_batch], precision=precision,
      preferred_element_type=None)
  if axis_name:
    out_tup = xla.parallel_translations[psum_p](
        c, local_out, axes=axis_name, axis_index_groups=None,
        axis_env=axis_env, platform=platform)
    out, = xla.xla_destructure(c, out_tup)
  else:
    out = local_out
  return out
xla.parallel_translations[pdot_p] = _pdot_translation_rule

def _pdot_transpose_lhs(g, y, *, axis_name, pos_contract, pos_batch, precision):
  # TODO: avals with names, call pbroadcast with axis_name
  return lax._dot_general_transpose_lhs(
      g, y, dimension_numbers=[pos_contract, pos_batch], precision=precision,
      preferred_element_type=None)
def _pdot_transpose_rhs(g, x, *, axis_name, pos_contract, pos_batch, precision):
  # TODO: avals with names, call pbroadcast with axis_name
  return lax._dot_general_transpose_rhs(
      g, x, dimension_numbers=[pos_contract, pos_batch], precision=precision,
      preferred_element_type=None)
ad.defbilinear(pdot_p, _pdot_transpose_lhs, _pdot_transpose_rhs)

pxla.multi_host_supported_collectives.add(pdot_p)


def _pgather_impl(src, idx, *, axes):
  assert all(isinstance(axis, int) for axis in axes)
  src_axes_front = moveaxis(src, axes, range(len(axes)))
  non_axes_shape = src_axes_front.shape[len(axes):]
  src_one_axis_front = src_axes_front.reshape((-1,) + non_axes_shape)
  slice_sizes = (1,) + non_axes_shape
  idx = lax.reshape(idx, idx.shape + (1,))
  offset_dims = tuple(range(idx.ndim - 1, idx.ndim + src_one_axis_front.ndim - 2))
  dnums = lax.GatherDimensionNumbers(
      offset_dims=offset_dims,
      collapsed_slice_dims=(0,),
      start_index_map=(0,))
  return lax.gather(src_one_axis_front, idx, dimension_numbers=dnums,
                    slice_sizes=tuple(slice_sizes))

def _pgather_abstract_eval(src, idx, *, axes):
  # TODO: Avals with names rule: remove all axes from src, insert those from idx
  #       The order is important, because it is ok to re-insert one of the deleted axes!
  shape = list(src.shape)
  for axis in sorted((a for a in axes if isinstance(a, int)), reverse=True):
    del shape[axis]
  shape = idx.shape + tuple(shape)
  return ShapedArray(shape, src.dtype)

def _pgather_parallel_translation(c, src, idx, *, axes, axis_env, platform):
  if any(not isinstance(axis, int) for axis in axes):
    raise NotImplementedError("pgather only supported in the SPMD lowering."
                              "Please open a feature request!")
  return xla.lower_fun(_pgather_impl, multiple_results=False)(c, src, idx, axes=axes)

def _pgather_batcher(vals_in, dims_in, *, axes):
  src, idx = vals_in
  dsrc, didx = dims_in
  if didx is not batching.not_mapped and dsrc is not batching.not_mapped:
    # NB: We could just go forward with it and take the diagonal along the
    #     two axes we get in the output, but that would be quite inefficient
    raise NotImplementedError("Please open a feature request!")
  elif didx is not batching.not_mapped:
    return pgather_p.bind(src, idx, axes=axes), didx
  elif dsrc is not batching.not_mapped:
    src_last_batched = moveaxis(src, dsrc, -1)
    result = pgather_p.bind(src_last_batched, idx, axes=axes)
    return result, result.ndim - 1
  else:
    assert False  # This shouldn't get called anyway

def _pgather_collective_batcher(frame, vals_in, dims_in, *, axes):
  src, idx = vals_in
  dsrc, didx = dims_in
  if dsrc is batching.not_mapped:
    raise ValueError("pgather axis {frame.name} is missing from the indexed value")
  if didx is not batching.not_mapped:
    # NOTE: This is allowed and the output would be mapped along this axis!
    raise NotImplementedError("Please open a feature request!")
  # Now source is mapped, idx is not
  new_axes = tuple(dsrc if axis == frame.name else
                   axis + (dsrc <= axis) if isinstance(axis, int) else
                   axis
                   for axis in axes)
  # The result is not mapped, because we eliminate all axes, and those include
  # the batched axis.
  if all(isinstance(axis, int) for axis in axes):
    # We rewrite a purely positional pgather as a gather, because that one
    # is more fully featured (e.g. supports AD).
    return _pgather_impl(src, idx, axes=new_axes), batching.not_mapped
  else:
    return pgather_p.bind(src, idx, axes=new_axes), batching.not_mapped

pgather_p = core.AxisPrimitive('pgather')
pgather_p.def_impl(_pgather_impl)
pgather_p.def_abstract_eval(_pgather_abstract_eval)
xla.parallel_translations[pgather_p] = _pgather_parallel_translation
# TODO: Transpose? That requires adding pscatter...
batching.primitive_batchers[pgather_p] = _pgather_batcher
batching.collective_rules[pgather_p] = _pgather_collective_batcher
core.axis_substitution_rules[pgather_p] = partial(_subst_all_names_in_param, 'axes')
