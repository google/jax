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

import collections

import numpy as np

from jax import core
from jax import dtypes
from jax import tree_util
from jax.lax import lax
from jax.abstract_arrays import ShapedArray, raise_to_shaped, ConcreteArray
from jax.interpreters import ad
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.interpreters import batching
from jax.interpreters import masking
from jax.util import partial, unzip2, prod, safe_map
from jax.lib import xla_client as xc
from jax.config import config
from jax.numpy import lax_numpy

from jax.interpreters.pxla import axis_index

map = safe_map

xops = xc.ops

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
      once, and all groups must be the same size.


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
  [ 0.          0.16666667  0.33333334  0.5       ]
  """
  _validate_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  leaves = [lax.convert_element_type(l, np.int32)
            if dtypes.dtype(l) == np.bool_ else l for l in leaves]
  out_flat = psum_p.bind(*leaves, axis_name=axis_name,
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
      once, and all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce mean along the axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [ 1.5         1.5         1.5         1.5       ]
  >>> y = jax.pmap(lambda x: x / jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [ 0.          0.66666667  1.33333334  2.0       ]
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
      once, and all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce max along the axis ``axis_name``.
  """
  _validate_axis_index_groups(axis_index_groups)
  return tree_util.tree_map(partial(
      pmax_p.bind, axis_name=axis_name, axis_index_groups=axis_index_groups), x)

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
      once, and all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce min along the axis ``axis_name``.
  """
  _validate_axis_index_groups(axis_index_groups)
  return tree_util.tree_map(partial(
      pmin_p.bind, axis_name=axis_name, axis_index_groups=axis_index_groups), x)

def _validate_axis_index_groups(axis_index_groups):
  if axis_index_groups is None:
    return
  len_0 = len(axis_index_groups[0])
  if any(len(g) != len_0 for g in axis_index_groups):
    raise ValueError("axis_index_groups must all be the same size")
  axis_space = range(len_0 * len(axis_index_groups))
  if set(i for g in axis_index_groups for i in g) != set(axis_space):
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


def pswapaxes(x, axis_name, axis):
  """Swap the pmapped axis ``axis_name`` with the unmapped axis ``axis``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  The mapped axis size must be equal to the size of the unmapped axis; that is,
  we must have ``lax.psum(1, axis_name) == x.shape[axis]``.

  This function is a special case of ``all_to_all`` where the pmapped axis of
  the input is placed at the position ``axis`` in the output. That is, it is
  equivalent to ``all_to_all(x, axis_name, axis, axis)``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.

  Returns:
    Array(s) with shape ``np.insert(np.delete(x.shape, axis), axis, axis_size)``
    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``.
  """
  return all_to_all(x, axis_name, axis, axis)

def all_to_all(x, axis_name, split_axis, concat_axis):
  """Materialize the mapped axis and map a different axis.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  In the output, the input mapped axis ``axis_name`` is materialized at the
  logical axis position ``concat_axis``, and the input unmapped axis at position
  ``split_axis`` is mapped with the name ``axis_name``.

  The input mapped axis size must be equal to the size of the axis to be mapped;
  that is, we must have ``lax.psum(1, axis_name) == x.shape[split_axis]``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    split_axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    concat_axis: int indicating the position in the output to materialize the
      mapped axis of the input with the name ``axis_name``.

  Returns:
    Array(s) with shape given by the expression::

      np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)

    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``, i.e. ``axis_size = lax.psum(1, axis_name)``.
  """
  def bind(x):
    if psum(1, axis_name) != x.shape[split_axis]:
      msg = ("all_to_all requires the size of the mapped axis axis_name to "
             "equal x.shape[split_axis], but they are {} and {} respectively.")
      raise ValueError(msg.format(psum(1, axis_name), x.shape[split_axis]))
    return all_to_all_p.bind(x, split_axis=split_axis, concat_axis=concat_axis,
                             axis_name=axis_name)
  return tree_util.tree_map(bind, x)

### parallel primitives

def _allreduce_soft_pmap_rule(prim, reducer, vals, mapped, chunk_size,
                              *, axis_name, axis_index_groups):
  if axis_index_groups is not None:
    raise NotImplementedError("soft_pmap does not yet support axis_index_groups")
  reduced_vals = [reducer(x, [0]) if m else x for x, m in zip(vals, mapped)]
  outs = prim.bind(*reduced_vals, axis_name=axis_name,
                   axis_index_groups=axis_index_groups)
  return outs, (False,) * len(vals)

def _allreduce_translation_rule(prim, c, val, *, axis_name, axis_index_groups,
                                axis_env, platform):
  replica_groups = _replica_groups(axis_env, axis_name, axis_index_groups)
  dtype = c.get_shape(val).numpy_dtype()
  scalar = ShapedArray((), dtype)
  computation = xla.primitive_subcomputation(prim, scalar, scalar)
  replica_groups_protos = xc.make_replica_groups(replica_groups)
  return xops.AllReduce(val, computation, replica_groups_protos, None, None)

# It is assumed that all collectives that use this rule are commutative
# and associative over axis names if they support tuples. That is,
# they have to satisfy:
#   collective(x, ('i', 'j')) == collective(x, ('j', 'i'))
#                             == collective(collective(x, 'j'), 'i')
def _split_axis_comm_assoc(primitive, split_name, args, params):
  axis_names = params['axis_name']
  assert isinstance(axis_names, tuple)
  if params['axis_index_groups'] is not None:
    raise NotImplementedError("axis_index_groups not supported in axis splitting. "
                              "Please open a feature request!")
  remaining_axes = list(axis_names)
  remaining_axes.remove(split_name)
  remaining_axes = tuple(remaining_axes)
  split_params = dict(params, axis_name=split_name)
  remain_params = dict(params, axis_name=remaining_axes)
  split_result = primitive.bind(*args, **split_params)
  return primitive.bind(*split_result, **remain_params)

# NB: This is only used for collectives that do not include the vmapped axis name,
#     which is why the rule is so simple. All other collectives go through split_axis.
def _collective_batcher(prim, args, dims, **params):
  return prim.bind(*args, **params), dims

def _batched_reduction_collective(prim, if_mapped, if_unmapped,
                                 vals_in, dims_in, axis_size,
                                 axis_name, axis_index_groups):
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not implemented in vmap collectives. "
                              "Please open a feature request!")
  vals_out = [if_mapped(v, d) if d is not batching.not_mapped else if_unmapped(v, axis_size)
              for v, d in zip(vals_in, dims_in)]
  dims_out = [batching.not_mapped] * len(vals_in)
  return vals_out, dims_out

def _replica_groups(axis_env, axis_name, axis_index_groups):
  replica_groups = xla.axis_groups(axis_env, axis_name)
  if axis_index_groups is not None:
    replica_groups = [[axis_group[i] for i in axis_index_group]
                      for axis_group in replica_groups
                      for axis_index_group in axis_index_groups]
  return replica_groups

# psum translation rule has special handling for complex dtypes
def _psum_translation_rule(c, *args, axis_name, axis_index_groups, axis_env,
                           platform):
  if platform in ("cpu", "tpu"):
    return _notuple_psum_translation_rule(c, *args, axis_name=axis_name,
                                          axis_index_groups=axis_index_groups,
                                          axis_env=axis_env, platform=platform)

  # XLA's tuple all-reduce doesn't support different dtypes in the same
  # allreduce. Instead, we perform once all-reduce for each argument input type.
  args_by_type = collections.defaultdict(lambda: ([], []))
  for i, arg in enumerate(args):
    indices, dtype_args = args_by_type[c.get_shape(arg).numpy_dtype()]
    indices.append(i)
    dtype_args.append(arg)

  # The outputs, in the original argument order.
  out = [None] * len(args)
  replica_groups = _replica_groups(axis_env, axis_name, axis_index_groups)
  replica_groups_protos = xc.make_replica_groups(replica_groups)
  for dtype, (indices, dtype_args) in sorted(args_by_type.items()):
    is_complex = dtypes.issubdtype(dtype, np.complexfloating)
    n = len(dtype_args)
    if is_complex:
      dtype_args = ([xops.Real(x) for x in dtype_args] +
                    [xops.Imag(x) for x in dtype_args])
    scalar = ShapedArray((), c.get_shape(dtype_args[0]).numpy_dtype())
    computation = xla.primitive_subcomputation(lax.add_p, scalar, scalar)
    all_reduce = xops.AllReduce(xops.Tuple(c, dtype_args), computation,
                                replica_groups_protos, None, None)
    if is_complex:
      xs = [xops.Complex(xops.GetTupleElement(all_reduce, i),
                         xops.GetTupleElement(all_reduce, n + i)) for i in range(n)]
    else:
      xs = [xops.GetTupleElement(all_reduce, i) for i in range(n)]
    for i, x in zip(indices, xs):
      out[i] = x
  return xops.Tuple(c, out)

# TODO(b/150476027): CPU doesn't support tuple all-reduce correctly. But
# fortunately we don't really need it in that case because CPU doesn't support
# cross-task communication either.
# TODO(b/155446630): An XLA:TPU optimization pass also doesn't support
# tuple all-reduce yet. Meanwhile, rely on deterministic compiler behavior.
def _notuple_psum_translation_rule(c, *args, axis_name, axis_env,
                                   axis_index_groups, platform):
  def _translate(val):
    psum = partial(_allreduce_translation_rule, lax.add_p, c,
                   axis_name=axis_name, axis_env=axis_env,
                   axis_index_groups=axis_index_groups, platform=platform)
    dtype = c.get_shape(val).numpy_dtype()
    if dtypes.issubdtype(dtype, np.complexfloating):
      return xops.Complex(psum(xops.Real(val)), psum(xops.Imag(val)))
    else:
      return psum(val)
  return xops.Tuple(c, list(map(_translate, args)))

def _psum_transpose_rule(cts, axis_name, axis_index_groups):
  nonzero_out_cts, treedef = tree_util.tree_flatten(cts)
  nonzero_in_cts = psum_p.bind(*nonzero_out_cts, axis_name=axis_name,
                               axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, nonzero_in_cts)

def _allreduce_masking_rule(prim, identity, args, logical_shapes, axis_name,
                            axis_index_groups):
  # TODO(j-towns): refine this check once we have named axes in avals
  if not all(isinstance(core.get_aval(d), ConcreteArray)
             for shape in logical_shapes for d in shape):
    raise ValueError('abstract logical shape detected, you may be attempting '
                     'to perform a parallel collective over a ragged batch, '
                     'which is not supported.')
  if axis_index_groups is not None:
    raise NotImplementedError('Masking rule not implemented for primitive {} '
                              'with axis_index_groups'.format(prim))
  if isinstance(axis_name, tuple):
    raise NotImplementedError('Masking rule not yet implemented for primitive '
                              '{} with tuple axis_name'.format(prim))
  try:
    _, axis_size_poly = core.axis_frame_poly(axis_name)
  except NameError:
    return prim.bind(*args, axis_name=axis_name,
                     axis_index_groups=axis_index_groups)
  axis_size_logical = masking.eval_poly_logical(axis_size_poly)
  _, axis_size_padded = core.axis_frame(axis_name)
  assert axis_size_padded == masking.eval_poly_padded(axis_size_poly)
  mask = lambda a: lax.select(axis_index(axis_name) < axis_size_logical, a,
                              lax.full_like(a, identity))
  return prim.bind(*map(mask, args), axis_name=axis_name,
                   axis_index_groups=axis_index_groups)

psum_p = core.Primitive('psum')
psum_p.multiple_results = True
psum_p.def_impl(partial(pxla.apply_parallel_primitive, psum_p))
psum_p.def_abstract_eval(lambda *args, **params: map(raise_to_shaped, args))
pxla.soft_pmap_rules[psum_p] = \
    partial(_allreduce_soft_pmap_rule, psum_p, lax._reduce_sum)
xla.parallel_translations[psum_p] = _psum_translation_rule
pxla.parallel_pure_rules[psum_p] = lambda *args, shape: (x * prod(shape) for x in args)
ad.deflinear(psum_p, _psum_transpose_rule)
pxla.multi_host_supported_collectives.add(psum_p)
batching.split_axis_rules[psum_p] = partial(_split_axis_comm_assoc, psum_p)
batching.primitive_batchers[psum_p] = partial(_collective_batcher, psum_p)
batching.collective_rules[psum_p] = \
  partial(_batched_reduction_collective,
          psum_p,
          lambda v, d: v.sum(d),
          lambda v, axis_size: axis_size * v)
masking.masking_rules[psum_p] = partial(_allreduce_masking_rule, psum_p, 0)

pmax_p = core.Primitive('pmax')
pmax_p.def_abstract_eval(lambda x, **params: raise_to_shaped(x))
xla.parallel_translations[pmax_p] = \
    partial(_allreduce_translation_rule, lax.max_p)
batching.split_axis_rules[pmax_p] = partial(_split_axis_comm_assoc, pmax_p)
batching.primitive_batchers[pmax_p] = partial(_collective_batcher, pmax_p)
batching.collective_rules[pmax_p] = \
  partial(_batched_reduction_collective,
          pmax_p,
          lambda v, d: v.max(d),
          lambda v, axis_size: v)
masking.masking_rules[pmax_p] = partial(_allreduce_masking_rule, pmax_p, -np.inf)

pmin_p = core.Primitive('pmin')
pmin_p.def_abstract_eval(lambda x, **params: raise_to_shaped(x))
xla.parallel_translations[pmin_p] = \
    partial(_allreduce_translation_rule, lax.min_p)
batching.split_axis_rules[pmin_p] = partial(_split_axis_comm_assoc, pmin_p)
batching.primitive_batchers[pmin_p] = partial(_collective_batcher, pmin_p)
batching.collective_rules[pmin_p] = \
  partial(_batched_reduction_collective,
          pmin_p,
          lambda v, d: v.min(d),
          lambda v, axis_size: v)
masking.masking_rules[pmin_p] = partial(_allreduce_masking_rule, pmin_p, np.inf)

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

def _ppermute_transpose_rule(t, perm, axis_name):
  srcs, dsts = unzip2(perm)
  inverse_perm = list(zip(dsts, srcs))
  return [ppermute(t, axis_name=axis_name, perm=inverse_perm)]

def _ppermute_batcher(vals_in, dims_in, axis_size, axis_name, perm):
  assert len(perm) == axis_size, "Permutation doesn't match the axis size!"
  perm_indices = np.full((axis_size,), -1, dtype=np.int32)
  for s, d in perm:
    perm_indices[s] = d
  vals_out = [lax_numpy.take(v, perm_indices, d) if d is not batching.not_mapped else v
              for v, d in zip(vals_in, dims_in)]
  return vals_out, dims_in

ppermute_p = core.Primitive('ppermute')
ppermute_p.def_abstract_eval(lambda x, **params: raise_to_shaped(x))
ad.deflinear(ppermute_p, _ppermute_transpose_rule)
xla.parallel_translations[ppermute_p] = _ppermute_translation_rule
pxla.multi_host_supported_collectives.add(ppermute_p)
batching.primitive_batchers[ppermute_p] = partial(_collective_batcher, pmin_p)
batching.collective_rules[ppermute_p] = _ppermute_batcher


def _all_to_all_translation_rule(c, x, *, split_axis, concat_axis, axis_name,
                                 axis_env, platform):
  # Workaround for AllToAll not being implemented on CPU.
  replica_groups = _replica_groups(axis_env, axis_name, None)
  if len(replica_groups[0]) == 1:
    return x
  else:
    split_count = len(replica_groups[0])
    if not all(split_count == len(g) for g in replica_groups):
      raise ValueError('Replica groups must be equally sized')
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    return xops.AllToAll(x, split_axis, concat_axis, split_count,
                         replica_groups_protos)

def _all_to_all_split_axis_rule(vals, which_mapped, split_axis, concat_axis,
                                axis_name):
  assert tuple(which_mapped) == (True,)
  x, = vals
  # perform the communication to swap the hardware-mapped axes
  stacked = all_to_all_p.bind(x, split_axis=split_axis + 1, concat_axis=0,
                              axis_name=axis_name)
  # transpose the newly mapped axis to the front, newly unmapped to concat_axis
  out = _moveaxis(split_axis + 1, 0, stacked)
  out = _moveaxis(1, concat_axis + 1, out)
  return out, True

def _all_to_all_transpose_rule(cts, axis_name, split_axis, concat_axis):
  return (all_to_all(cts, axis_name=axis_name, split_axis=concat_axis, concat_axis=split_axis),)

def _moveaxis(src, dst, x):
  perm = [i for i in range(x.ndim) if i != src]
  perm.insert(dst, src)
  return lax.transpose(x, perm)

all_to_all_p = core.Primitive('all_to_all')
all_to_all_p.def_abstract_eval(lambda x, **params: raise_to_shaped(x))
xla.parallel_translations[all_to_all_p] = _all_to_all_translation_rule
ad.deflinear(all_to_all_p, _all_to_all_transpose_rule)
pxla.multi_host_supported_collectives.add(all_to_all_p)


def _expand(dim, size, axis_name, x):
  shape = list(x.shape)
  shape.insert(dim, size)
  out = lax.full(shape, lax._const(x, 0))
  return lax.dynamic_update_index_in_dim(out, x, axis_index(axis_name), dim)

def _allgather(x, dim, size, axis_name):
  outs = tree_util.tree_map(partial(_expand, dim, size, axis_name), x)
  return psum(outs, axis_name)

def all_gather(x, axis_name):
  """Gather values of x across all replicas.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This is equivalent to, but faster than, all_to_all(broadcast(x)).

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).

  Returns:
    Array(s) representing the result of an all-gather along the axis
    ``axis_name``. Shapes are the same as ``x.shape``, but with a leading
    dimension of the axis_size.

  For example, with 2 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.all_gather(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [[0 1 2 3]
   [0 1 2 3]
   [0 1 2 3]
   [0 1 2 3]]
  """
  return _allgather(x, 0, psum(1, axis_name), axis_name)


@config.omnistaging_enablers.append
def omnistaging_enabler() -> None:
  # We set a special bind rule for psum so that psum(1, 'i') can be evaluated at
  # tracing time.
  @psum_p.def_custom_bind
  def psum_bind(*args, axis_name, axis_index_groups):
    if all(not isinstance(x, core.Tracer) for x in args):
      if axis_index_groups is not None:
        size = len(axis_index_groups[0])
      elif type(axis_name) is tuple:
        size = prod([core.axis_frame(name).size for name in axis_name])  # type: ignore
      else:
        size = core.axis_frame(axis_name).size  # type: ignore
      return tuple(size * x for x in args)
    return core.Primitive.bind(
        psum_p, *args, axis_name=axis_name, axis_index_groups=axis_index_groups)
