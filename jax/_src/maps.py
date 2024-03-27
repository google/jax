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

from collections import OrderedDict, abc
from collections.abc import Iterable, Sequence, Mapping
import contextlib
from functools import wraps, partial, partialmethod, lru_cache
import itertools as it
import math
from typing import Callable, Any, NamedTuple, Union

import numpy as np

from jax import lax
from jax import numpy as jnp

from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import effects
from jax._src import mesh as mesh_lib
from jax._src import linear_util as lu
from jax._src import op_shardings
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import stages
from jax._src import traceback_util
from jax._src.api_util import (flatten_fun_nokwargs, flatten_axes,
                               _ensure_index_tuple, donation_vector,
                               shaped_abstractify, check_callable)
from jax._src.array import ArrayImpl
from jax._src.errors import JAXTypeError
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import xla
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters.partial_eval import (
  trace_to_subjaxpr_dynamic, DynamicJaxprTracer,
  convert_constvars_jaxpr, new_jaxpr_eqn)
from jax._src.interpreters import pxla
from jax._src.pjit import (sharding_constraint_p, get_unconstrained_dims,
                           GSPMDSharding)
from jax._src.sharding_impls import (
    ArrayMapping, NamedSharding, ParsedPartitionSpec,
    array_mapping_to_axis_resources, UNSPECIFIED)
from jax._src.tree_util import (tree_flatten, tree_unflatten, all_leaves,
                                tree_map, treedef_tuple)
from jax._src.util import (safe_map, safe_zip, HashableFunction, unzip2, unzip3,
                           as_hashable_function, distributed_debug_log,
                           tuple_insert, moveaxis, split_list, wrap_name,
                           merge_lists, partition_list)

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)

map, unsafe_map = safe_map, map
zip = safe_zip


class FrozenDict(abc.Mapping):
  def __init__(self, *args, **kwargs):
    self.contents = dict(*args, **kwargs)

  def __iter__(self):
    return iter(self.contents)

  def __len__(self):
    return len(self.contents)

  def __getitem__(self, name):
    return self.contents[name]

  def __eq__(self, other):
    return isinstance(other, FrozenDict) and self.contents == other.contents

  def __hash__(self):
    return hash(tuple(self.contents.items()))

  def __repr__(self):
    return f"FrozenDict({self.contents})"

# Multi-dimensional generalized map

AxisName = core.AxisName
ResourceAxisName = mesh_lib.ResourceAxisName  # Different name just for documentation purposes
Mesh = mesh_lib.Mesh
MeshAxisName = mesh_lib.MeshAxisName
ResourceEnv = mesh_lib.ResourceEnv
EMPTY_ENV = mesh_lib.EMPTY_ENV
thread_resources = mesh_lib.thread_resources


class SerialLoop:
  """Create an anonymous serial loop resource for use in a single xmap axis.

  A use of :py:class:`SerialLoop` in :py:func:`xmap`'s ``axis_resources``
  extends the resource environment with a new serial loop with a unique
  unspecified name, that will only be used to partition the axis that
  used a given instance.

  This is unlike :py:func:`serial_loop`, which makes it possible to iterate
  jointly over chunks of multiple axes (with the usual requirement that they
  do not coincide in a named shape of any value in the program).

  Example::

      # Processes `x` in a vectorized way, but in 20 micro-batches.
      xmap(f, in_axes=['i'], out_axes=[i], axis_resources={'i': SerialLoop(20)})(x)

      # Computes the result in a vectorized way, but in 400 micro-batches,
      # once for each coordinate (0, 0) <= (i, j) < (20, 20). Each `SerialLoop`
      # creates a fresh anonymous loop.
      xmap(h, in_axes=(['i'], ['j']), out_axes=['i', 'j'],
           axis_resources={'i': SerialLoop(20), 'j': SerialLoop(20)})(x, y)
  """
  length: int

  def __init__(self, length):
    self.length = length

  def __eq__(self, other):
    return self.length == other.length

  def __hash__(self):
    return hash(self.length)


@contextlib.contextmanager
def serial_loop(name: ResourceAxisName, length: int):
  """Define a serial loop resource to be available in scope of this context manager.

  This is similar to :py:class:`Mesh` in that it extends the resource
  environment with a resource called ``name``. But, any use of this resource
  axis in ``axis_resources`` argument of :py:func:`xmap` will cause the
  body of :py:func:`xmap` to get executed ``length`` times with each execution
  only processing only a slice of inputs mapped along logical axes assigned
  to this resource.

  This is especially useful in that it makes it possible to lower the memory
  usage compared to :py:func:`vmap`, because it will avoid simultaneous
  materialization of intermediate values for every point in the batch.

  Note that collectives over loop axes are not supported, so they are less
  versatile than physical mesh axes.

  Args:
    name: Name of the loop in the resource environment.
    length: Number of iterations.

  Example::

    >>> x = jnp.linspace(0, jnp.pi, 4)
    ...
    >>> with serial_loop('l', len(x)):
    ...   out = xmap(
    ...     lambda x: jnp.sin(x) * 5,  # This will be called 4 times with different
    ...                                # slices of x.
    ...     in_axes=['i'], out_axes=['i'],
    ...     axis_resources={'i': 'l'})(x)
    >>> out.shape
    (4,)
  """
  old_env: ResourceEnv = getattr(thread_resources, "env", EMPTY_ENV)
  thread_resources.env = old_env.with_extra_loop(mesh_lib.Loop(name, length))
  try:
    yield
  finally:
    thread_resources.env = old_env


_next_resource_id = 0
class _UniqueResourceName:
  def __init__(self, uid, tag=None):
    self.uid = uid
    self.tag = tag
  def __eq__(self, other):
    return type(other) is _UniqueResourceName and self.uid == other.uid
  def __hash__(self):
    return hash(self.uid)
  def __repr__(self):
    return f"<UniqueResource {self.tag} {self.uid}>"

def fresh_resource_name(tag=None):
  global _next_resource_id
  try:
    return _UniqueResourceName(_next_resource_id, tag)
  finally:
    _next_resource_id += 1


# This is really a Dict[AxisName, int], but we don't define a
# pytree instance for it, so that it is treated as a leaf.
class AxisNamePos(FrozenDict):
  user_repr: str
  expected_rank: int | None = None

  def __init__(self, *args, user_repr, **kwargs):
    super().__init__(*args, **kwargs)
    self.user_repr = user_repr

class AxisNamePosWithRank(AxisNamePos):
  def __init__(self, *args, expected_rank, **kwargs):
    super().__init__(*args, **kwargs)
    self.expected_rank = expected_rank


# str(...) == 'Ellipsis' which is really annoying
class DotDotDotRepr:
  def __repr__(self): return '...'


def _parse_entry(arg_name, entry):
  # Dictionaries mapping axis names to positional axes
  if isinstance(entry, dict) and all(isinstance(v, int) for v in entry.keys()):
    result = AxisNamePos(((name, axis) for axis, name in entry.items()),
                         user_repr=str(entry))
    num_mapped_dims = len(entry)
  # Non-empty lists or tuples that optionally terminate with an ellipsis
  elif isinstance(entry, (tuple, list)):
    if entry and entry[-1] == ...:
      constr = AxisNamePos
      entry = entry[:-1]
      tail = [DotDotDotRepr()] if isinstance(entry, list) else (DotDotDotRepr(),)
      user_repr = str(entry + tail)
    else:
      constr = partial(AxisNamePosWithRank, expected_rank=len(entry))
      user_repr = str(entry)
    result = constr(((name, axis) for axis, name in enumerate(entry)
                     if name is not None),
                    user_repr=user_repr)
    num_mapped_dims = sum(name is not None for name in entry)
  else:
    raise TypeError(f"""\
Value mapping specification in xmap {arg_name} pytree can be either:
- lists of axis names (possibly ending with the ellipsis object: ...)
- dictionaries that map positional axes (integers) to axis names (e.g. {{2: 'name'}})
but got: {entry}""")
  if len(result) != num_mapped_dims:
    raise ValueError(f"Named axes should be unique within each {arg_name} argument "
                     f"specification, but one them is: {entry}")
  for axis in result.values():
    if axis < 0:
      raise ValueError(f"xmap doesn't support negative axes in {arg_name}")
  return result

def _is_axes_leaf(entry):
  if isinstance(entry, dict) and all_leaves(entry.values()):
    return True
  # NOTE: `None`s are not considered leaves by `all_leaves`
  if isinstance(entry, (tuple, list)) and all_leaves(v for v in entry if v is not None):
    return True
  return False

def _prepare_axes(axes, arg_name):
  entries, treedef = tree_flatten(axes, is_leaf=_is_axes_leaf)
  entries = map(partial(_parse_entry, arg_name), entries)
  return tree_unflatten(treedef, entries), entries, treedef

Resource = Union[ResourceAxisName, SerialLoop]
ResourceSet = Union[Resource, tuple[Resource, ...]]

# TODO: Some syntactic sugar to make the API more usable in a single-axis case?
# TODO: Are the resource axes scoped lexically or dynamically? Dynamically for now!
def xmap(fun: Callable,
         in_axes,
         out_axes,
         *,
         axis_sizes: Mapping[AxisName, int] | None = None,
         axis_resources: Mapping[AxisName, ResourceSet] | None = None,
         donate_argnums: int | Sequence[int] = (),
         backend: str | None = None) -> stages.Wrapped:
  """Assign a positional signature to a program that uses named array axes.

  .. warning::
    xmap is deprecated and will be removed in a future release. Use
    :py:func:`~jax.shard_map` or :py:func:`~jax.vmap` with the
    ``spmd_axis_name`` argument for expressing SPMD device-parallel
    computations. Please file an issue on https://github.com/google/jax/issues
    if neither are suitable for your use case.

  .. warning::
    This is an experimental feature and the details can change at
    any time. Use at your own risk!

  .. warning::
    This docstring is aspirational. Not all features of the named axis
    programming model have been implemented just yet.

  The usual programming model of JAX (or really NumPy) associates each array
  with two pieces of metadata describing its type: the element type (``dtype``)
  and the ``shape``. :py:func:`xmap` extends this model by adding support for
  *named axes*.  In particular, each array used in a function wrapped by
  :py:func:`xmap` can additionally have a non-empty ``named_shape`` attribute,
  which can be used to query the set of named axes (introduced by
  :py:func:`xmap`) appearing in that value along with their shapes.
  Furthermore, in most places where positional axis indices are allowed (for
  example the `axes` arguments in :py:func:`sum`), bound axis names are also
  accepted. The :py:func:`einsum` language is extended inside :py:func:`xmap`
  to additionally allow contractions that involve named axes.  Broadcasting of
  named axes happens *by name*, i.e. all axes with equal names are expected to
  have equal shapes in all arguments of a broadcasting operation, while the
  result has a (set) union of all named axes.  The positional semantics of the
  program remain unchanged, and broadcasting still implicitly right-aligns
  positional axes for unification. For an extended description of the
  :py:func:`xmap` programming model, please refer to the :py:func:`xmap`
  tutorial notebook in main JAX documentation.

  Note that since all top-level JAX expressions are interpreted in the NumPy
  programming model, :py:func:`xmap` can also be seen as an adapter that
  converts a function that uses named axes (including in arguments and returned
  values) into one that takes and returns values that only have positional
  axes.

  The default lowering strategy of :py:func:`xmap` converts all named axes into
  positional axes, working similarly to multiple applications of
  :py:func:`vmap`. However, this behavior can be further customized by the
  ``axis_resources`` argument.  When specified, each axis introduced by
  :py:func:`xmap` can be assigned to one or more *resource axes*. Those include
  the axes of the hardware mesh, as defined by the :py:class:`Mesh` context
  manager. Each value that has a named axis in its ``named_shape`` will be
  partitioned over all mesh axes that axis is assigned to. Hence,
  :py:func:`xmap` can be seen as an alternative to :py:func:`pmap` that also
  exposes a way to automatically partition the computation over multiple
  devices.

  .. warning::
    While it is possible to assign multiple axis names to a single resource axis,
    care has to be taken to ensure that none of those named axes co-occur in a
    ``named_shape`` of any value in the named program. At the moment this is
    **completely unchecked** and will result in **undefined behavior**. The
    final release of :py:func:`xmap` will enforce this invariant, but it is a
    work in progress.

    Note that you do not have to worry about any of this for as long as no
    resource axis is repeated in ``axis_resources.values()``.

  Note that any assignment of ``axis_resources`` doesn't ever change the
  results of the computation, but only how it is carried out (e.g. how many
  devices are used).  This makes it easy to try out various ways of
  partitioning a single program in many distributed scenarios (both small- and
  large-scale), to maximize the performance.  As such, :py:func:`xmap` can be
  seen as a way to seamlessly interpolate between :py:func:`vmap` and
  :py:func:`pmap`-style execution.

  Args:
    fun: Function that uses named axes. Its arguments and return
      value should be arrays, scalars, or (nested) standard Python containers
      (tuple/list/dict) thereof (in general: valid pytrees).
    in_axes: A Python object with the same container (pytree) structure as the
      signature of arguments to ``fun``, but with a positional-to-named axis
      mapping in place of every array argument. The valid positional-to-named
      mappings are: (1) a ``Dict[int, AxisName]`` specifying that a positional
      dimensions given by dictionary keys are to be converted to named axes
      of given names (2) a list of axis names that ends with the Ellipsis object
      (``...``) in which case a number of leading positional axes of the argument
      will be converted into named axes inside the function. Note that ``in_axes``
      can also be a prefix of the argument container structure, in which case the
      mapping is repeated for all arrays in the collapsed subtree.
    out_axes: A Python object with the same container (pytree) structure as the
      returns of ``fun``, but with a positional-to-named axis mapping in place
      of every returned array. The valid positional-to-named mappings are the same
      as in ``in_axes``. Note that ``out_axes`` can also be a prefix of the return
      container structure, in which case the mapping is repeated for all arrays
      in the collapsed subtree.
    axis_sizes: A dict mapping axis names to their sizes. All axes defined by xmap
      have to appear either in ``in_axes`` or ``axis_sizes``. Sizes of axes
      that appear in ``in_axes`` are inferred from arguments whenever possible.
      In multi-host scenarios, the user-specified sizes are expected to be the
      global axis sizes (and might not match the expected size of local inputs).
    axis_resources: A dictionary mapping the axes introduced in this
      :py:func:`xmap` to one or more resource axes. Any array that has in its
      shape an axis with some resources assigned will be partitioned over the
      resources associated with the respective resource axes.
    donate_argnums: Specify which argument buffers are "donated" to the computation.
      It is safe to donate argument buffers if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to.

      For more details on buffer donation see the `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.

    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend. 'cpu', 'gpu', or 'tpu'.

  Returns:
    A version of ``fun`` that takes in arrays with positional axes in place of
    named axes bound in this :py:func:`xmap` call, and results with all named
    axes converted to positional axes. If ``axis_resources`` is specified,
    ``fun`` can additionally execute in parallel on multiple devices.

  For example, :py:func:`xmap` makes it very easy to convert a function that
  computes the vector inner product (such as :py:func:`jax.numpy.vdot`) into
  one that computes a matrix multiplication:

  >>> import jax.numpy as jnp
  >>> x = jnp.arange(10).reshape((2, 5))
  >>> xmap(jnp.vdot,
  ...      in_axes=({0: 'left'}, {1: 'right'}),
  ...      out_axes=['left', 'right', ...])(x, x.T)
  Array([[ 30,  80],
         [ 80, 255]], dtype=int32)

  Note that the contraction in the program is performed over the positional axes,
  while named axes are just a convenient way to achieve batching. While this
  might seem like a silly example at first, it might turn out to be useful in
  practice, since with conjunction with ``axis_resources`` this makes it possible
  to implement a distributed matrix-multiplication in just a few lines of code::

    devices = np.array(jax.devices())[:4].reshape((2, 2))
    with Mesh(devices, ('x', 'y')):  # declare a 2D mesh with axes 'x' and 'y'
      distributed_out = xmap(
        jnp.vdot,
        in_axes=({0: 'left'}, {1: 'right'}),
        out_axes=['left', 'right', ...],
        axis_resources={'left': 'x', 'right': 'y'})(x, x.T)

  Still, the above examples are quite simple. After all, the xmapped
  computation was a simple NumPy function that didn't use the axis names at all!
  So, let's explore a slightly larger example which is linear regression::

    def regression_loss(x, y, w, b):
      # Contract over in_features. Batch and out_features are present in
      # both inputs and output, so they don't need to be mentioned
      y_pred = jnp.einsum('{in_features},{in_features}->{}', x, w) + b
      error = jnp.sum((y - y_pred) ** 2, axis='out_features')
      return jnp.mean(error, axis='batch')

    xmap(regression_loss,
         in_axes=(['batch', 'in_features', ...],
                  ['batch', 'out_features', ...],
                  ['in_features', 'out_features', ...],
                  ['out_features', ...]),
         out_axes={})  # Loss is reduced over all axes, including batch!

  .. note::
    When using ``axis_resources`` along with a mesh that is controlled by
    multiple JAX hosts, keep in mind that in any given process :py:func:`xmap`
    only expects the data slice that corresponds to its local devices to be
    specified. This is in line with the current multi-host :py:func:`pmap`
    programming model.
  """
  check_callable(fun)

  if isinstance(in_axes, list) and not _is_axes_leaf(in_axes):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/google/jax/issues/2367
    in_axes = tuple(in_axes)

  if in_axes == ():  # Allow empty argument lists
    in_axes, in_axes_entries = (), []
  else:
    in_axes, in_axes_entries, _ = _prepare_axes(in_axes, "in_axes")
  if out_axes == ():
    raise ValueError("xmapped functions cannot have no return values")
  else:
    out_axes, out_axes_entries, out_axes_treedef = _prepare_axes(out_axes, "out_axes")
    out_axes_entries = tuple(out_axes_entries)  # Make entries hashable

  axis_sizes = {} if axis_sizes is None else axis_sizes
  axis_resources = {} if axis_resources is None else axis_resources

  axis_sizes_names = set(axis_sizes.keys())
  in_axes_names = set(it.chain(*(spec.keys() for spec in in_axes_entries)))
  defined_names = axis_sizes_names | in_axes_names
  out_axes_names = set(it.chain(*(spec.keys() for spec in out_axes_entries)))

  anon_serial_loops = []
  def normalize_resource(r) -> ResourceAxisName:
    if isinstance(r, SerialLoop):
      name = fresh_resource_name()
      anon_serial_loops.append((name, r.length))
      return name
    return r

  axes_with_resources = set(axis_resources.keys())
  if axes_with_resources - defined_names:
    raise ValueError(f"All axes that were assigned resources have to appear in "
                     f"in_axes or axis_sizes, but the following are missing: "
                     f"{axes_with_resources - defined_names}")
  if out_axes_names - defined_names:
    raise ValueError(f"All axis names appearing in out_axes must also appear in "
                     f"in_axes or axis_sizes, but the following are missing: "
                     f"{out_axes_names - defined_names}")

  normalized_axis_resources: dict[AxisName, tuple[ResourceAxisName, ...]] = {}
  for axis in defined_names:
    resources = axis_resources.get(axis, ())
    if not isinstance(resources, tuple):
      resources = (resources,)
    normalized_axis_resources[axis] = tuple(unsafe_map(normalize_resource, resources))
  frozen_axis_resources = FrozenDict(normalized_axis_resources)
  necessary_resources = set(it.chain(*frozen_axis_resources.values()))

  for axis, resources in frozen_axis_resources.items():
    if len(set(resources)) != len(resources):  # type: ignore
      raise ValueError(f"Resource assignment of a single axis must be a tuple of "
                       f"distinct resources, but specified {resources} for axis {axis}")

  donate_argnums = _ensure_index_tuple(donate_argnums)

  # A little performance optimization to avoid iterating over all args unnecessarily
  has_input_rank_assertions = any(spec.expected_rank is not None for spec in in_axes_entries)
  has_output_rank_assertions = any(spec.expected_rank is not None for spec in out_axes_entries)

  def infer_params(*args):
    # Putting this outside of fun_mapped would make resources lexically scoped
    resource_env = thread_resources.env
    available_resources = set(resource_env.shape.keys())

    if necessary_resources - available_resources:
      raise ValueError(f"In-scope resources are insufficient to execute the "
                       f"xmapped function. The missing resources are: "
                       f"{necessary_resources - available_resources}")

    args_flat, in_tree = tree_flatten(args)
    fun_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    if donate_argnums:
      donated_invars = donation_vector(donate_argnums, (), args, {})
    else:
      donated_invars = (False,) * len(args_flat)
    in_axes_flat = _flatten_axes("xmap in_axes", in_tree, in_axes, tupled_args=True)

    # Some pytree containers might be unhashable, so we flatten the out_axes
    # pytree into a treedef and entries which are guaranteed to be hashable.
    out_axes_thunk = HashableFunction(
      lambda: tuple(_flatten_axes("xmap out_axes", out_tree(), out_axes, tupled_args=False)),
      closure=(out_axes_entries, out_axes_treedef))

    axis_resource_count = _get_axis_resource_count(
        frozen_axis_resources, resource_env)

    for axis, size in axis_sizes.items():
      resources = axis_resource_count[axis]
      if size % resources.nglobal != 0:
        global_size = "Global size" if resources.distributed else "Size"
        raise ValueError(f"{global_size} of axis {axis} ({size}) is not divisible "
                         f"by the total number of resources assigned to this axis "
                         f"({frozen_axis_resources[axis]}, {resources.nglobal} in total)")
    frozen_global_axis_sizes = _get_axis_sizes(
        args_flat, in_axes_flat, axis_sizes, axis_resource_count)

    missing_sizes = defined_names - set(frozen_global_axis_sizes.keys())
    if missing_sizes:
      raise ValueError(f"Failed to infer size of axes: {', '.join(unsafe_map(str, missing_sizes))}. "
                       f"You've probably passed in empty containers in place of arguments that had "
                       f"those axes in their in_axes. Provide the sizes of missing axes explicitly "
                       f"via axis_sizes to fix this error.")

    if has_input_rank_assertions:
      for arg, spec in zip(args_flat, in_axes_flat):
        if spec.expected_rank is not None and spec.expected_rank != arg.ndim:
          raise ValueError(f"xmap argument has an in_axes specification of {spec.user_repr}, "
                           f"which asserts that it should be of rank {spec.expected_rank}, "
                           f"but the argument has rank {arg.ndim} (and shape {arg.shape})")

    _check_gda_or_array_xmap_partitioning(
        frozen_axis_resources, resource_env, frozen_global_axis_sizes,
        in_axes_flat, args_flat)

    params = dict(
      name=getattr(fun, '__name__', '<unnamed function>'),
      in_axes=tuple(in_axes_flat),
      out_axes_thunk=out_axes_thunk,
      donated_invars=donated_invars,
      global_axis_sizes=frozen_global_axis_sizes,
      axis_resources=frozen_axis_resources,
      resource_env=resource_env,
      backend=backend,
      spmd_in_axes=None,
      spmd_out_axes_thunk=None)
    return fun_flat, args_flat, params, in_tree, out_tree

  def verify_outputs(out_flat, out_tree, params):
    if has_output_rank_assertions:
      for out, spec in zip(out_flat, params['out_axes_thunk']()):
        if spec.expected_rank is not None and spec.expected_rank != out.ndim:
          raise ValueError(f"xmap output has an out_axes specification of {spec.user_repr}, "
                           f"which asserts that it should be of rank {spec.expected_rank}, "
                           f"but the output has rank {out.ndim} (and shape {out.shape})")
    return tree_unflatten(out_tree(), out_flat)

  def decorate_serial(f):
    for loop_params in reversed(anon_serial_loops):
      f = serial_loop(*loop_params)(f)
    return f

  @wraps(fun)
  @decorate_serial
  def fun_mapped(*args):
    tree_map(dispatch.check_arg, args)
    fun_flat, args_flat, params, _, out_tree = infer_params(*args)
    out_flat = xmap_p.bind(fun_flat, *args_flat, **params)
    return verify_outputs(out_flat, out_tree, params)

  @decorate_serial
  def lower(*args, **kwargs):
    lowering_parameters = kwargs.pop(
        '_experimental_lowering_platform', mlir.LoweringParameters())
    fun_flat, args_flat, params, in_tree, out_tree = infer_params(*args)
    avals_flat = [shaped_abstractify(arg) for arg in args_flat]
    computation = make_xmap_callable(
        fun_flat, params['name'], params['in_axes'], params['out_axes_thunk'],
        params['donated_invars'], params['global_axis_sizes'], params['axis_resources'],
        params['resource_env'], params['backend'], params['spmd_in_axes'],
        params['spmd_out_axes_thunk'],
        lowering_parameters, *avals_flat)

    in_tree = treedef_tuple([in_tree, tree_flatten({})[1]])
    in_avals = in_tree.unflatten(avals_flat)
    return stages.Lowered.from_flat_info(
        computation, in_tree, in_avals, donate_argnums, out_tree(),  # type: ignore
        no_kwargs=True)

  fun_mapped.lower = lower
  return fun_mapped

def xmap_impl(fun: lu.WrappedFun, *args, name, in_axes, out_axes_thunk, donated_invars,
              global_axis_sizes, axis_resources, resource_env, backend,
              spmd_in_axes, spmd_out_axes_thunk):
  in_avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  xmap_callable = make_xmap_callable(
      fun, name, in_axes, out_axes_thunk, donated_invars, global_axis_sizes,
      axis_resources, resource_env, backend,
      spmd_in_axes, spmd_out_axes_thunk,
      mlir.LoweringParameters(), *in_avals).compile().unsafe_call
  distributed_debug_log(("Running xmapped function", name),
                        ("python function", fun.f),
                        ("mesh", resource_env.physical_mesh),
                        ("abstract args", in_avals))
  return xmap_callable(*args)

@lu.cache
def make_xmap_callable(fun: lu.WrappedFun,
                       name,
                       in_axes, out_axes_thunk, donated_invars,
                       global_axis_sizes, axis_resources, resource_env, backend,
                       spmd_in_axes, spmd_out_axes_thunk,
                       lowering_parameters: mlir.LoweringParameters,
                       *in_avals):
  plan = EvaluationPlan.from_axis_resources(
      axis_resources, resource_env, global_axis_sizes)

  # TODO: Making axis substitution final style would allow us to avoid
  #       tracing to jaxpr here
  mapped_in_avals = [_delete_aval_axes(aval, in_axes, global_axis_sizes)
                     for aval, in_axes in zip(in_avals, in_axes)]
  with core.extend_axis_env_nd(global_axis_sizes.items()):
    with dispatch.log_elapsed_time(
        "Finished tracing + transforming {fun_name} for xmap in {elapsed_time} sec",
        fun_name=fun.__name__, event=dispatch.JAXPR_TRACE_EVENT):
      jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, mapped_in_avals)
  out_axes = out_axes_thunk()
  # _check_out_avals_vs_out_axes(out_avals, out_axes, global_axis_sizes)
  # NOTE: We don't use avals and all params, so only pass in the relevant parts (too lazy...)
  _resource_typing_xmap([], dict(axis_resources=axis_resources,
                                 out_axes=out_axes,
                                 call_jaxpr=jaxpr,
                                 resource_env=resource_env,
                                 name=name),
                        source_info_util.new_source_info(), resource_env, {})
  jaxpr = plan.subst_axes_with_resources(jaxpr)
  use_spmd_lowering = SPMD_LOWERING.value
  ensure_fixed_sharding = _ENSURE_FIXED_SHARDING.value
  if use_spmd_lowering and ensure_fixed_sharding:
    jaxpr = _fix_inferred_spmd_sharding(jaxpr, resource_env)

  f = lu.wrap_init(core.jaxpr_as_fun(core.ClosedJaxpr(jaxpr, consts)))
  f = hide_mapped_axes(f, tuple(in_axes), tuple(out_axes))
  f = plan.vectorize_and_loop(f, in_axes, out_axes)

  used_resources = _jaxpr_resources(jaxpr, resource_env) | set(it.chain(*axis_resources.values()))
  used_mesh_axes = used_resources & resource_env.physical_resource_axes
  if used_mesh_axes:
    assert spmd_in_axes is None and spmd_out_axes_thunk is None  # No outer xmaps, so should be None
    mesh_in_axes, mesh_out_axes = plan.to_mesh_axes(in_axes, out_axes)
    mesh = resource_env.physical_mesh
    tiling_method: pxla.TilingMethod
    if SPMD_LOWERING_MANUAL.value:
      manual_mesh_axes = frozenset(it.chain.from_iterable(plan.physical_axis_resources.values()))
      tiling_method = pxla.TileManual(manual_mesh_axes)
    else:
      tiling_method = pxla.TileVectorize()
    in_shardings = [NamedSharding(mesh, array_mapping_to_axis_resources(i))
                    for i in mesh_in_axes]
    out_shardings = [NamedSharding(mesh, array_mapping_to_axis_resources(o))
                     for o in mesh_out_axes]
    return pxla.lower_mesh_computation(
        f, 'xmap', name, mesh,
        in_shardings, out_shardings, donated_invars,
        use_spmd_lowering, in_avals,
        tiling_method=tiling_method,
        lowering_parameters=lowering_parameters)
  else:
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(f, in_avals)
    return pxla.lower_sharding_computation(
        core.ClosedJaxpr(jaxpr, consts), 'jit', name,
        (UNSPECIFIED,) * len(in_avals), (UNSPECIFIED,) * len(out_avals),
        donated_invars, in_avals, keep_unused=True,
        devices_from_context=None, lowering_parameters=lowering_parameters,
        in_layouts=(None,) * len(in_avals), out_layouts=(None,) * len(out_avals))


class EvaluationPlan(NamedTuple):
  """Encapsulates preprocessing common to top-level xmap invocations and its translation rule."""
  resource_env: ResourceEnv
  physical_axis_resources: dict[AxisName, tuple[ResourceAxisName, ...]]
  loop_axis_resources: dict[AxisName, tuple[ResourceAxisName, ...]]
  axis_subst_dict: dict[AxisName, tuple[ResourceAxisName, ...]]
  axis_vmap_size: dict[AxisName, int | None]

  @property
  def axis_subst(self) -> core.AxisSubst:
    return lambda name: self.axis_subst_dict.get(name, (name,))

  @property
  def resource_axis_env(self):
    env = dict(self.resource_env.shape)
    for axis, size in self.axis_vmap_size.items():
      if size is None:
        continue
      vmap_axis = self.axis_subst_dict[axis][-1]
      env[vmap_axis] = size
    return env

  @classmethod
  def from_axis_resources(cls,
                          axis_resources: dict[AxisName, tuple[ResourceAxisName, ...]],
                          resource_env: ResourceEnv,
                          global_axis_sizes: dict[AxisName, int]):
    physical_axis_resources, loop_axis_resources = _unzip_axis_resources(
            axis_resources, resource_env)
    axis_resource_count = _get_axis_resource_count(
        axis_resources, resource_env)
    axis_subst_dict = dict(axis_resources)
    axis_vmap_size: dict[AxisName, int | None] = {}
    for naxis, raxes in sorted(axis_resources.items(), key=lambda x: str(x[0])):
      num_resources = axis_resource_count[naxis]
      assert global_axis_sizes[naxis] % num_resources.nglobal == 0
      local_tile_size = global_axis_sizes[naxis] // num_resources.nglobal
      # We have to vmap when there are no resources (to handle the axis name!) or
      # when every resource gets chunks of values.
      if not raxes or local_tile_size > 1:
        axis_vmap_size[naxis] = local_tile_size
        axis_subst_dict[naxis] += (fresh_resource_name(naxis),)
      else:
        axis_vmap_size[naxis] = None
    return cls(resource_env,
               physical_axis_resources, loop_axis_resources,
               axis_subst_dict, axis_vmap_size)

  def subst_axes_with_resources(self, jaxpr):
    try:
      if any(self.loop_axis_resources.values()):
        _check_no_loop_collectives(jaxpr, self.loop_axis_resources)
      with core.extend_axis_env_nd(self.resource_axis_env.items()):
        return core.subst_axis_names_jaxpr(jaxpr, self.axis_subst)
    except core.DuplicateAxisNameError:
      raise AssertionError("Incomplete resource type-checking? Please open a bug report!")

  def vectorize_and_loop(self, f: lu.WrappedFun, in_axes, out_axes) -> lu.WrappedFun:
    vmap_axes = {
        naxis: raxes[-1]
        for naxis, raxes in self.axis_subst_dict.items()
        if self.axis_vmap_size[naxis] is not None
    }
    for naxis, vaxis in sorted(vmap_axes.items(), key=lambda x: x[1].uid):
      local_tile_size = self.axis_vmap_size[naxis]
      map_in_axes = tuple(unsafe_map(lambda spec: spec.get(naxis, None), in_axes))
      map_out_axes = tuple(unsafe_map(lambda spec: spec.get(naxis, None), out_axes))
      f = batching.vtile(f, map_in_axes, map_out_axes, tile_size=local_tile_size, axis_name=vaxis)

    used_loops = set(it.chain.from_iterable(self.loop_axis_resources.values()))
    if not used_loops:
      return f

    if len(used_loops) > 1:
      # TODO: Support multiple loops
      raise NotImplementedError("Only one loop per xmap is supported")
    loop_in_axes = _to_resource_axes(in_axes, self.loop_axis_resources)
    loop_out_axes = _to_resource_axes(out_axes, self.loop_axis_resources)
    loop_name, = used_loops
    loop_length = self.resource_env.shape[loop_name]
    def looped_f(*args):
      def body(i, _):
        # XXX: This call_wrapped is only valid under the assumption that scan
        #      only ever traces the body once (which it does at the moment).
        result = f.call_wrapped(
            *(_slice_tile(arg, spec.get(loop_name, None), i, loop_length)
              for arg, spec in zip(args, loop_in_axes)))
        return i + 1, result
      _, stacked_results = lax.scan(body, 0, (), length=loop_length)
      return [_merge_leading_axis(sresult, spec.get(loop_name, None))
              for sresult, spec in zip(stacked_results, loop_out_axes)]
    return lu.wrap_init(looped_f)

  def to_mesh_axes(self, in_axes, out_axes=None):
    """
    Convert in/out_axes parameters ranging over logical dimensions to
    in/out_axes that range over the mesh dimensions.
    """
    if out_axes is None:
      return _to_resource_axes(in_axes, self.physical_axis_resources)
    else:
      return (_to_resource_axes(in_axes, self.physical_axis_resources),
              _to_resource_axes(out_axes, self.physical_axis_resources))

# -------- xmap primitive and its transforms --------

# xmap has a different set of parameters than pmap, so we make it its own primitive type
class XMapPrimitive(core.MapPrimitive):
  def __init__(self):
    super().__init__('xmap')
    self.def_impl(xmap_impl)
    self.def_custom_bind(self.bind)

  def bind(self, fun, *args, in_axes, **params):
    assert len(in_axes) == len(args), (in_axes, args)
    return core.map_bind(self, fun, *args, in_axes=in_axes, **params)

  def process(self, trace, fun, tracers, params):
    return trace.process_xmap(self, fun, tracers, params)

  def post_process(self, trace, out_tracers, params):
    post_process = getattr(trace, 'post_process_xmap', None)
    if post_process is None:
      raise NotImplementedError
    return post_process(self, out_tracers, params)

  def get_bind_params(self, params):
    new_params = dict(params)
    jaxpr = new_params.pop('call_jaxpr')
    subfun = lu.hashable_partial(lu.wrap_init(core.eval_jaxpr), jaxpr, ())
    axes = new_params.pop('out_axes')
    new_params['out_axes_thunk'] = HashableFunction(lambda: axes, closure=axes)
    spmd_axes = new_params.pop('spmd_out_axes')
    if spmd_axes is not None:
      new_params['spmd_out_axes_thunk'] = \
          HashableFunction(lambda: spmd_axes, closure=spmd_axes)
    else:
      new_params['spmd_out_axes_thunk'] = None
    return [subfun], new_params

xmap_p = XMapPrimitive()
core.EvalTrace.process_xmap = core.EvalTrace.process_call  # type: ignore
def _process_xmap_default(self, call_primitive, f, tracers, params):
  raise NotImplementedError(f"{type(self)} must override process_xmap to handle xmap")
core.Trace.process_xmap = _process_xmap_default  # type: ignore

def _xmap_axis_subst(params, subst, traverse):
  if 'call_jaxpr' not in params:  # TODO(apaszke): This feels sketchy, but I'm not sure why
    return params
  if not traverse:
    return params
  def shadowed_subst(name):
    return (name,) if name in params['global_axis_sizes'] else subst(name)
  with core.extend_axis_env_nd(params['global_axis_sizes'].items()):
    new_jaxpr = core.subst_axis_names_jaxpr(params['call_jaxpr'], shadowed_subst)
  return dict(params, call_jaxpr=new_jaxpr)
core.axis_substitution_rules[xmap_p] = _xmap_axis_subst

# NOTE: We don't have to handle spmd_{in|out}_axes here, because
# SPMD batching always gets involved as the last transform before XLA translation
ad.JVPTrace.process_xmap = ad.JVPTrace.process_call  # type: ignore
ad.call_param_updaters[xmap_p] = pxla.xla_call_jvp_update_params

def _xmap_transpose(params, call_jaxpr, args, cts_in, cts_in_avals):
  all_args, in_tree_def = tree_flatten(((), args, cts_in))  # empty consts
  fun = lu.hashable_partial(lu.wrap_init(ad.backward_pass), call_jaxpr, False)
  fun, nz_arg_cts = ad.nonzero_outputs(fun)
  fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
  # Preserve axis for primal arguments, skip tangents (represented as undefined primals).
  in_axes, out_axes = params['in_axes'], params['out_axes']
  new_in_axes = (*(axis for axis, x in zip(in_axes, args) if not ad.is_undefined_primal(x)),
                 *(axis for axis, x in zip(out_axes, cts_in) if type(x) is not ad.Zero))
  # NOTE: This assumes that the output cotangents being zero is a deterministic
  #       function of which input cotangents were zero.
  @as_hashable_function(closure=(in_axes, tuple(type(c) is ad.Zero for c in cts_in)))
  def out_axes_thunk():
    return tuple(axis for axis, nz in zip(in_axes, nz_arg_cts()) if nz)
  new_params = dict(params,
                    name=wrap_name(params['name'], 'transpose'),
                    in_axes=new_in_axes,
                    out_axes_thunk=out_axes_thunk,
                    donated_invars=(False,) * len(new_in_axes),
                    spmd_out_axes_thunk=None)
  del new_params['out_axes']
  del new_params['spmd_out_axes']
  out_flat = xmap_p.bind(fun, *all_args, **new_params)
  arg_cts = tree_unflatten(out_tree(), out_flat)

  axis_resource_count = _get_axis_resource_count(
      params['axis_resources'], params['resource_env'])
  local_axis_sizes = {
      axis: axis_resource_count[axis].to_local(global_size)
      for axis, global_size in params['global_axis_sizes'].items()
  }
  def unmap_zero(zero, axes):
    return ad.Zero(_insert_aval_axes(zero.aval, axes, local_axis_sizes))
  return tuple(unmap_zero(arg_ct, in_axis) if type(arg_ct) is ad.Zero else arg_ct
               for arg_ct, in_axis in zip(arg_cts, in_axes))
ad.primitive_transposes[xmap_p] = _xmap_transpose


def _typecheck_xmap(
    _, *in_atoms, call_jaxpr, name, in_axes, out_axes, donated_invars,
    global_axis_sizes, axis_resources, resource_env, backend,
    spmd_in_axes, spmd_out_axes):
  in_avals = [x.aval for x in in_atoms]
  axis_resource_count = _get_axis_resource_count(
      axis_resources, resource_env)
  local_axis_sizes = {
      axis: axis_resource_count[axis].to_local(global_size)
      for axis, global_size in global_axis_sizes.items()
  }
  binder_in_avals = [_insert_aval_axes(v.aval, a_in_axes, local_axis_sizes)
                     for v, a_in_axes in zip(call_jaxpr.invars, in_axes)]
  for binder_in_aval, in_aval in zip(binder_in_avals, in_avals):
    if not core.typecompat(binder_in_aval, in_aval):
      raise core.JaxprTypeError(
        f"xmap passes operand {in_aval} to jaxpr expecting {binder_in_aval}")

  with core.extend_axis_env_nd(global_axis_sizes.items()):
    core._check_jaxpr(lambda: core.JaxprPpContext(), call_jaxpr)

  mapped_out_avals = [v.aval for v in call_jaxpr.outvars]
  out_avals = [_insert_aval_axes(a, a_out_axes, local_axis_sizes)
               for a, a_out_axes in zip(mapped_out_avals, out_axes)]
  effs = core.filter_named_axis_effects(call_jaxpr.effects, global_axis_sizes)
  return out_avals, effs
core.custom_typechecks[xmap_p] = _typecheck_xmap


def _resource_typing_xmap(avals,
                          params,
                          source_info: source_info_util.SourceInfo,
                          resource_env,
                          outer_axis_resources):
  axis_resources = params['axis_resources']
  inner_axis_resources = dict(outer_axis_resources)
  inner_axis_resources.update(axis_resources)
  if len(inner_axis_resources) < len(outer_axis_resources) + len(axis_resources):
    overlap = set(outer_axis_resources) & set(axis_resources)
    raise JAXTypeError(
        f"Detected disallowed xmap axis name shadowing at "
        f"{source_info_util.summarize(source_info)} "
        f"(shadowed axes: {mesh_lib.show_axes(overlap)})")

  if resource_env.physical_mesh != params['resource_env'].physical_mesh:
    raise RuntimeError("Changing the physical mesh is not allowed inside xmap.")

  call_jaxpr = params['call_jaxpr']
  pxla.resource_typecheck(
      params['call_jaxpr'], resource_env, inner_axis_resources,
      lambda: (f"an xmapped function {params['name']} " +
               (f"(xmap called at {source_info_util.summarize(source_info)})"
                if source_info else "")))

  for v, axes in zip(call_jaxpr.outvars, params['out_axes']):
    broadcast_axes = set(axes) - set(v.aval.named_shape)
    used_resources = set(it.chain.from_iterable(
        inner_axis_resources[a] for a in v.aval.named_shape))
    for baxis in broadcast_axes:
      baxis_resources = set(inner_axis_resources[baxis])
      overlap = baxis_resources & used_resources
      if overlap:
        resource_to_axis = {}
        for axis in v.aval.named_shape:
          for raxis in inner_axis_resources[axis]:
            resource_to_axis[raxis] = axis
        partitioning_axes = {resource_to_axis[raxis] for raxis in overlap}
        raise JAXTypeError(
            f"One of xmapped function ({params['name']}) outputs is broadcast "
            f"along axis `{baxis}` which is assigned to resources "
            f"{mesh_lib.show_axes(baxis_resources)}, but the output is already "
            f"partitioned along {mesh_lib.show_axes(overlap)}, because its "
            f"named shape contains {mesh_lib.show_axes(partitioning_axes)}")
pxla.custom_resource_typing_rules[xmap_p] = _resource_typing_xmap


# This is DynamicJaxprTrace.process_map with some very minor modifications
def _dynamic_jaxpr_process_xmap(self, primitive, f, tracers, params):
  assert primitive is xmap_p
  in_avals = [t.aval for t in tracers]
  global_axis_sizes = params['global_axis_sizes']
  mapped_in_avals = [_delete_aval_axes(a, a_in_axes, global_axis_sizes)
                     for a, a_in_axes in zip(in_avals, params['in_axes'])]
  with core.extend_axis_env_nd(global_axis_sizes.items()):
    with core.new_sublevel():
      jaxpr, mapped_out_avals, consts, () = trace_to_subjaxpr_dynamic(
          f, self.main, mapped_in_avals)
  out_axes = params['out_axes_thunk']()
  if params['spmd_out_axes_thunk'] is not None:
    spmd_out_axes = params['spmd_out_axes_thunk']()
  else:
    spmd_out_axes = None
  axis_resource_count = _get_axis_resource_count(
      params['axis_resources'], params['resource_env'])
  local_axis_sizes = {
      axis: axis_resource_count[axis].to_local(global_size)
      for axis, global_size in global_axis_sizes.items()
  }
  out_avals = [_insert_aval_axes(a, a_out_axes, local_axis_sizes)
               for a, a_out_axes in zip(mapped_out_avals, out_axes)]
  # _check_out_avals_vs_out_axes(out_avals, out_axes, params['global_axis_sizes'])
  source_info = source_info_util.current()
  out_tracers = [DynamicJaxprTracer(self, a, source_info) for a in out_avals]
  invars = map(self.getvar, tracers)
  constvars = map(self.getvar, map(self.instantiate_const, consts))
  outvars = map(self.makevar, out_tracers)
  new_in_axes = (AxisNamePos(user_repr='{}'),) * len(consts) + params['in_axes']
  if params['spmd_in_axes'] is None:
    new_spmd_in_axes = None
  else:
    new_spmd_in_axes = (None,) * len(consts) + params['spmd_in_axes']
  new_donated_invars = (False,) * len(consts) + params['donated_invars']
  with core.extend_axis_env_nd(global_axis_sizes.items()):
    call_jaxpr = convert_constvars_jaxpr(jaxpr)
  new_params = dict(params, in_axes=new_in_axes, out_axes=out_axes,
                    donated_invars=new_donated_invars,
                    spmd_in_axes=new_spmd_in_axes,
                    spmd_out_axes=spmd_out_axes,
                    call_jaxpr=call_jaxpr)
  del new_params['out_axes_thunk']
  del new_params['spmd_out_axes_thunk']
  effs = core.filter_named_axis_effects(call_jaxpr.effects, global_axis_sizes)
  eqn = new_jaxpr_eqn([*constvars, *invars], outvars, primitive,
                      new_params, effs, source_info)
  self.frame.add_eqn(eqn)
  return out_tracers
pe.DynamicJaxprTrace.process_xmap = _dynamic_jaxpr_process_xmap  # type: ignore

def _xmap_partial_eval_custom_params_updater(
    unks_in: Sequence[bool], inst_in: Sequence[bool],
    kept_outs_known: Sequence[bool], kept_outs_staged: Sequence[bool],
    num_res: int, params_known: dict, params_staged: dict
  ) -> tuple[dict, dict]:
  assert params_known['spmd_in_axes'] is None is params_known['spmd_out_axes']
  assert params_staged['spmd_in_axes'] is None is params_staged['spmd_out_axes']

  # prune inputs to jaxpr_known according to unks_in
  donated_invars_known, _ = pe.partition_list(unks_in, params_known['donated_invars'])
  in_axes_known, _ = pe.partition_list(unks_in, params_known['in_axes'])
  if num_res == 0:
    residual_axes = []
  else:
    residual_axes = [
      AxisNamePos(zip(sort_named_shape, range(len(sort_named_shape))),
                  user_repr=f'<internal: {sort_named_shape}>')
      for named_shape in (v.aval.named_shape for v in params_known['call_jaxpr'].outvars[:-num_res])
      # We sort here to make the iteration order deterministic
      for sort_named_shape in [sorted(named_shape, key=str)]
    ]
  _, out_axes_known = pe.partition_list(kept_outs_known, params_known['out_axes'])
  new_params_known = dict(params_known,
                          in_axes=tuple(in_axes_known),
                          out_axes=(*out_axes_known, *residual_axes),
                          donated_invars=tuple(donated_invars_known))
  assert len(new_params_known['in_axes']) == len(params_known['call_jaxpr'].invars)
  assert len(new_params_known['out_axes']) == len(params_known['call_jaxpr'].outvars)

  # added num_res new inputs to jaxpr_staged, and pruning according to inst_in
  _, donated_invars_staged = pe.partition_list(inst_in, params_staged['donated_invars'])
  donated_invars_staged = [False] * num_res + donated_invars_staged
  _, in_axes_staged = pe.partition_list(inst_in, params_staged['in_axes'])
  in_axes_staged = [*residual_axes, *in_axes_staged]
  _, out_axes_staged = pe.partition_list(kept_outs_staged, params_staged['out_axes'])
  new_params_staged = dict(params_staged, in_axes=tuple(in_axes_staged),
                           out_axes=tuple(out_axes_staged),
                           donated_invars=tuple(donated_invars_staged))
  assert len(new_params_staged['in_axes']) == len(params_staged['call_jaxpr'].invars)
  assert len(new_params_staged['out_axes']) == len(params_staged['call_jaxpr'].outvars)
  return new_params_known, new_params_staged
pe.partial_eval_jaxpr_custom_rules[xmap_p] = \
    partial(pe.call_partial_eval_custom_rule, 'call_jaxpr',
            _xmap_partial_eval_custom_params_updater)


@lu.transformation_with_aux
def out_local_named_shapes(local_axes, *args, **kwargs):
  ans = yield args, kwargs
  ans_axes = [frozenset(a.aval.named_shape) & local_axes for a in ans]
  yield ans, ans_axes


def _jaxpr_trace_process_xmap(self, primitive, f: lu.WrappedFun, tracers, params):
  assert primitive is xmap_p
  assert params['spmd_out_axes_thunk'] is params['spmd_in_axes'] is None
  in_axes = params['in_axes']
  donated_invars = params['donated_invars']
  global_axis_sizes = params['global_axis_sizes']
  out_axes_thunk = params['out_axes_thunk']

  # Adjust input tracers' pvals for mapped axes, and unpack.
  in_pvals = [t.pval if t.pval.is_known() else
              pe.PartialVal.unknown(
                  _delete_aval_axes(t.pval.get_aval(), axes, global_axis_sizes))
              for t, axes in zip(tracers, in_axes)]
  in_knowns, in_avals, in_consts = pe.partition_pvals(in_pvals)

  # Wrap f to perform partial evaluation, and plumb out aux data.
  f = pe.trace_to_subjaxpr_nounits(f, self.main, False)
  f, aux = pe.partial_eval_wrapper_nounits(f, tuple(in_knowns), tuple(in_avals))
  # Also grab the local named shapes of the output (known and res).
  f, out_named_shapes = out_local_named_shapes(f, frozenset(global_axis_sizes))

  # Adjust params for knowns (donated_invars, in_axes, out_axes_thunk).
  out_axes = None  # cache this to avoid calling out_axes_thunk() more than once

  @as_hashable_function(closure=out_axes_thunk)
  def new_out_axes_thunk():
    nonlocal out_axes
    out_axes = out_axes_thunk()
    out_knowns, _, _, _ = aux()
    _, out_axes_known = partition_list(out_knowns, out_axes)
    return (*out_axes_known, *res_axes())
  def res_axes():
    _, _, jaxpr_unknown, _ = aux()
    num_res = len(jaxpr_unknown.constvars)
    res_named_shapes = out_named_shapes()[-num_res:] if num_res else []
    sorted_named_shapes = [sorted(ns, key=str) for ns in res_named_shapes]
    return [AxisNamePos(zip(named_shape, range(len(named_shape))),
                        user_repr=f'<internal: {named_shape}>')
            for named_shape in sorted_named_shapes]
  known_params = dict(
      params, in_axes=tuple(a for a, k in zip(in_axes, in_knowns) if k),
      donated_invars=tuple(d for d, k in zip(donated_invars, in_knowns) if k),
      out_axes_thunk=new_out_axes_thunk)

  # Run the known part.
  out = primitive.bind(f, *in_consts, **known_params)
  out_knowns, out_avals, jaxpr_unknown, env = aux()
  known_outvals, res = split_list(out, [len(out)-len(jaxpr_unknown.constvars)])
  with core.extend_axis_env_nd(global_axis_sizes.items()):
    jaxpr_unknown = pe.convert_constvars_jaxpr(jaxpr_unknown)

  # Set up new params.
  if out_axes is None:
    out_axes = out_axes_thunk()  # new_out_axes_thunk may have set during bind
  out_axes_unknown = [a for a, k in zip(out_axes, out_knowns) if not k]
  unknown_params = dict(
      params, call_jaxpr=jaxpr_unknown, out_axes=tuple(out_axes_unknown),
      spmd_out_axes=None,
      donated_invars=(*(False for _ in res),
                      *(d for d, k in zip(donated_invars, in_knowns) if not k)),
      in_axes=(*res_axes(), *(None for _ in env),
               *(a for a, k in zip(in_axes, in_knowns) if not k)))
  del unknown_params['out_axes_thunk']
  del unknown_params['spmd_out_axes_thunk']
  # Create input tracers for unknown part.
  res_tracers = map(self.new_instantiated_const, res)
  env_tracers = map(self.full_raise, env)
  unknown_arg_tracers = [t for t in tracers if not t.pval.is_known()]
  # Create output tracers for unknown part, adjusting avals.
  axis_resource_count = _get_axis_resource_count(
      params['axis_resources'], params['resource_env'])
  local_axis_sizes = {
      ax: axis_resource_count[ax].to_local(global_size)
      for ax, global_size in global_axis_sizes.items()}
  out_pvals = [pe.PartialVal.unknown(_insert_aval_axes(a, ax, local_axis_sizes))
               for a, ax in zip(out_avals, out_axes_unknown)]
  unknown_tracers_out = [pe.JaxprTracer(self, pval, None) for pval in out_pvals]
  # Build eqn to be staged out and attach it to unknown output tracers.
  eqn = pe.new_eqn_recipe((*res_tracers, *env_tracers, *unknown_arg_tracers),
                          unknown_tracers_out, primitive, unknown_params,
                          jaxpr_unknown.effects, source_info_util.current())
  for t in unknown_tracers_out: t.recipe = eqn
  return merge_lists(out_knowns, unknown_tracers_out, known_outvals)
pe.JaxprTrace.process_xmap = _jaxpr_trace_process_xmap

def _batch_trace_update_spmd_axes(
    spmd_in_axes, spmd_out_axes_thunk,
    axis_name, dims, dims_out_thunk):
  """Extends spmd in and out axes with the position of the trace's batch dimension."""
  not_mapped = batching.not_mapped
  def insert_spmd_axis(axes, nd):
    too_short = nd - len(axes)
    if too_short > 0:
      axes += (None,) * too_short
    return tuple_insert(axes, nd, axis_name)

  if spmd_in_axes is None:
    spmd_in_axes = ((),) * len(dims)
  new_spmd_in_axes = tuple(
    spmd_axes if d is not_mapped else insert_spmd_axis(spmd_axes, d)
    for spmd_axes, d in zip(spmd_in_axes, dims))

  @as_hashable_function(closure=spmd_out_axes_thunk)
  def new_spmd_out_axes_thunk():
    dims_out = dims_out_thunk()
    if spmd_out_axes_thunk is None:
      spmd_out_axes = ((),) * len(dims_out)
    else:
      spmd_out_axes = spmd_out_axes_thunk()
    return tuple(
      spmd_out_axes if nd is not_mapped else insert_spmd_axis(spmd_out_axes, nd)
      for spmd_out_axes, nd in zip(spmd_out_axes, dims_out))

  return new_spmd_in_axes, new_spmd_out_axes_thunk

def _axis_after_insertion(axis, inserted_named_axes):
  for inserted_axis in sorted(inserted_named_axes.values()):
    if inserted_axis >= axis:
      break
    axis += 1
  return axis

def _fmap_dims(axes, f):
  return AxisNamePos(((name, f(axis)) for name, axis in axes.items()),
                     user_repr=axes.user_repr)

def _batch_trace_process_xmap(self, is_spmd, primitive, f: lu.WrappedFun, tracers, params):
  not_mapped = batching.not_mapped
  vals, dims = unzip2((t.val, t.batch_dim) for t in tracers)
  assert primitive is xmap_p
  if not is_spmd and all(dim is not_mapped for dim in dims):
    return primitive.bind(f, *vals, **params)
  else:
    assert len({x.shape[d] for x, d in zip(vals, dims) if d is not not_mapped}) == 1
    new_in_axes = tuple(
      _fmap_dims(in_axes, lambda a: a + (d is not not_mapped and d <= a))
      for d, in_axes in zip(dims, params['in_axes']))
    mapped_dims_in = tuple(
      d if d is not_mapped else d - sum(a < d for a in in_axis.values())
      for d, in_axis in zip(dims, params['in_axes']))
    f, mapped_dims_out = batching.batch_subtrace(f, self.main, mapped_dims_in)
    out_axes_thunk: Callable[[], Sequence[AxisNamePos]] = params['out_axes_thunk']
    dims_out_thunk = lambda: tuple(d if d is not_mapped else _axis_after_insertion(d, out_axes)
                                   for d, out_axes in zip(mapped_dims_out(), out_axes_thunk()))
    # NOTE: This assumes that the choice of the dimensions over which outputs
    #       are batched is entirely dependent on the function and not e.g. on the
    #       data or its shapes.
    @as_hashable_function(closure=out_axes_thunk)
    def new_out_axes_thunk():
      return tuple(
        out_axes if d is not_mapped else
        _fmap_dims(out_axes, lambda a, nd=_axis_after_insertion(d, out_axes): a + (nd <= a))
        for out_axes, d in zip(out_axes_thunk(), mapped_dims_out()))

    if not is_spmd:
      assert params['spmd_in_axes'] is None and params['spmd_out_axes_thunk'] is None
      new_spmd_in_axes = None
      new_spmd_out_axes_thunk = None
    else:
      new_spmd_in_axes, new_spmd_out_axes_thunk = _batch_trace_update_spmd_axes(
        params['spmd_in_axes'], params['spmd_out_axes_thunk'],
        self.axis_name, dims, dims_out_thunk)

    new_params = dict(params,
                      in_axes=new_in_axes, out_axes_thunk=new_out_axes_thunk,
                      spmd_in_axes=new_spmd_in_axes,
                      spmd_out_axes_thunk=new_spmd_out_axes_thunk)
    vals_out = primitive.bind(f, *vals, **new_params)
    dims_out = dims_out_thunk()
    return [batching.BatchTracer(self, v, d) for v, d in zip(vals_out, dims_out)]
batching.BatchTrace.process_xmap = partialmethod(_batch_trace_process_xmap, False)  # type: ignore
pxla.SPMDBatchTrace.process_xmap = partialmethod(_batch_trace_process_xmap, True)  # type: ignore


def _batch_trace_post_process_xmap(self, primitive, out_tracers, params):
  not_mapped = batching.not_mapped
  BT = batching.BatchTracer
  vals, dims, srcs = unzip3((t.val, t.batch_dim, t.source_info) for t in out_tracers)
  main = self.main
  def todo(vals):
    trace = main.with_cur_sublevel()
    return [BT(trace, v, d if d is not_mapped else _axis_after_insertion(d, oa), s)
            for v, d, oa, s in zip(vals, dims, params['out_axes_thunk'](), srcs)]
  def out_axes_transform(out_axes):
    return tuple(oa if d is not_mapped else
                 _fmap_dims(oa, lambda a, nd=_axis_after_insertion(d, oa): a + (nd <= a))
                 for oa, d in zip(out_axes, dims))
  return vals, (todo, out_axes_transform)
batching.BatchTrace.post_process_xmap = _batch_trace_post_process_xmap


# -------- nested xmap handling --------

def _xmap_lowering_rule(ctx, *args, **kwargs):
  if isinstance(ctx.module_context.axis_context, sharding_impls.SPMDAxisContext):
    if SPMD_LOWERING_MANUAL.value:
      return _xmap_lowering_rule_spmd_manual(ctx, *args, **kwargs)
    else:
      return _xmap_lowering_rule_spmd(ctx, *args, **kwargs)
  # Here ShardingContext is used in place of ReplicaAxisContext because when
  # axis_resources and mesh is not used with xmap, `make_xmap_callable` will
  # go via `dispatch.sharded_lowering` path which sets the context to
  # ShardingContext. sharding_impls.ShardingContext is not used for SPMD.
  elif isinstance(ctx.module_context.axis_context,
                  (sharding_impls.ReplicaAxisContext, sharding_impls.ShardingContext)):
    return _xmap_lowering_rule_replica(ctx, *args, **kwargs)
  else:
    raise AssertionError("Unrecognized axis context type!")
mlir.register_lowering(xmap_p, _xmap_lowering_rule)

def _xmap_lowering_rule_replica(ctx, *in_nodes,
                                call_jaxpr, name,
                                in_axes, out_axes, donated_invars,
                                global_axis_sizes,
                                spmd_in_axes, spmd_out_axes,
                                axis_resources, resource_env, backend):
  mlir.check_backend_matches(backend, ctx.module_context.platforms)
  del backend, donated_invars
  # The only way for any of those two assertions to be violated is when xmap
  # is using the SPMD lowering, but then this rule shouldn't even trigger.
  assert spmd_in_axes is None and spmd_out_axes is None
  plan = EvaluationPlan.from_axis_resources(
      axis_resources, resource_env, global_axis_sizes)

  axis_resource_count = _get_axis_resource_count(
      axis_resources, resource_env)
  if any(resource_count.distributed for resource_count in axis_resource_count.values()):
    raise NotImplementedError

  mesh = resource_env.physical_mesh
  mesh_in_axes, mesh_out_axes = plan.to_mesh_axes(in_axes, out_axes)

  local_avals = [pxla.tile_aval_nd(
                    mesh.shape, aval_mesh_in_axes,
                    _insert_aval_axes(v.aval, aval_in_axes, global_axis_sizes))
                 for v, aval_in_axes, aval_mesh_in_axes
                 in zip(call_jaxpr.invars, in_axes, mesh_in_axes)]
  # We have to substitute before tracing, because we want the vectorized
  # axes to be used in the jaxpr.
  resource_call_jaxpr = plan.subst_axes_with_resources(call_jaxpr)
  f = lu.wrap_init(core.jaxpr_as_fun(core.ClosedJaxpr(resource_call_jaxpr, ())))
  f = hide_mapped_axes(f, tuple(in_axes), tuple(out_axes))
  f = plan.vectorize_and_loop(f, in_axes, out_axes)
  # NOTE: We don't extend the resource env with the mesh shape, because those
  #       resources are already in scope! It's the outermost xmap that introduces
  #       them!
  vectorized_jaxpr, out_avals, consts, () = pe.trace_to_jaxpr_dynamic(f, local_avals)
  # _check_out_avals_vs_out_axes(out_avals, out_axes, global_axis_sizes)
  const_nodes = [mlir.ir_constants(xla.canonicalize_dtype(x)) for x in consts]

  local_mesh_shape = mesh.local_mesh.shape
  tiled_ins = (
    mlir.lower_fun(partial(_tile, in_axes=arg_in_axes,
                           axis_sizes=local_mesh_shape),
                   multiple_results=False)(
          ctx.replace(primitive=None,
                      avals_in=[aval], avals_out=None),
          in_node)[0]
    for v, aval, in_node, arg_in_axes
    in zip(call_jaxpr.invars, ctx.avals_in, in_nodes, mesh_in_axes))

  # NOTE: We don't extend the resource env with the mesh shape, because those
  #       resources are already in scope! It's the outermost xmap that introduces
  #       them!
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  name_stack = ctx.name_stack.extend(wrap_name(name, 'xmap'))
  if any(effects.ordered_effects.contains(eff) for eff
         in vectorized_jaxpr.effects):
    raise NotImplementedError('Cannot lower `xmap` with ordered effects.')
  tiled_outs, _ = mlir.jaxpr_subcomp(ctx.module_context, vectorized_jaxpr,
                                     name_stack, mlir.TokenSet(),
                                     const_nodes, *tiled_ins,
                                     dim_var_values=ctx.dim_var_values)

  outs = [
      mlir.lower_fun(
          partial(_untile, out_axes=ans_out_axes, axis_sizes=local_mesh_shape),
          multiple_results=False)(
              ctx.replace(primitive=None,
                          avals_in=[vectorized_outvar.aval],
                          avals_out=None), tiled_out)[0]
      for v, vectorized_outvar, tiled_out, ans_out_axes
      in zip(call_jaxpr.outvars, vectorized_jaxpr.outvars, tiled_outs,
             mesh_out_axes)]
  return outs


def _xmap_lowering_rule_spmd(ctx, *global_in_nodes,
                             call_jaxpr, name, in_axes, out_axes,
                             donated_invars, global_axis_sizes, spmd_in_axes,
                             spmd_out_axes, axis_resources,
                             resource_env, backend):
  mlir.check_backend_matches(backend, ctx.module_context.platforms)
  del backend, donated_invars
  plan = EvaluationPlan.from_axis_resources(
      axis_resources, resource_env, global_axis_sizes)

  resource_call_jaxpr = plan.subst_axes_with_resources(call_jaxpr)
  f = lu.wrap_init(core.jaxpr_as_fun(core.ClosedJaxpr(resource_call_jaxpr, ())))
  f = hide_mapped_axes(f, in_axes, out_axes)
  f = plan.vectorize_and_loop(f, in_axes, out_axes)
  mesh_in_axes, mesh_out_axes = plan.to_mesh_axes(in_axes, out_axes)
  mesh = resource_env.physical_mesh
  f = pxla.vtile_by_mesh(f, mesh, mesh_in_axes, mesh_out_axes)

  # XXX: We modify mesh_in_axes and mesh_out_axes here
  def add_spmd_axes(
      flat_mesh_axes: Sequence[ArrayMapping],
      flat_extra_axes: Sequence[Sequence[Sequence[MeshAxisName]]] | None):
    if flat_extra_axes is None:
      return
    for axes, extra in zip(flat_mesh_axes, flat_extra_axes):
      for dim, dim_extra_axis in enumerate(extra):
        if dim_extra_axis is None: continue
        assert dim_extra_axis not in axes
        assert not config.enable_checks.value or all(v != dim for v in axes.values())
        axes[dim_extra_axis] = dim
  add_spmd_axes(mesh_in_axes, spmd_in_axes)
  add_spmd_axes(mesh_out_axes, spmd_out_axes)
  global_in_avals = ctx.avals_in
  vectorized_jaxpr, global_out_avals, consts, () = pe.trace_to_jaxpr_dynamic(f, global_in_avals)

  sharded_global_in_nodes = [
    [mlir.wrap_with_sharding_op(
        ctx, node, aval,
        NamedSharding(mesh, array_mapping_to_axis_resources(aval_axes)
                      )._to_xla_hlo_sharding(aval.ndim).to_proto())]
    if aval_axes else [node]
    for node, aval, aval_axes in zip(global_in_nodes, global_in_avals, mesh_in_axes)
  ]
  const_nodes = [mlir.ir_constants(xla.canonicalize_dtype(x)) for x in consts]

  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  name_stack = ctx.name_stack.extend(wrap_name(name, 'xmap'))
  if any(effects.ordered_effects.contains(eff) for eff
         in vectorized_jaxpr.effects):
    raise NotImplementedError('Cannot lower `xmap` with ordered effects.')
  global_out_nodes, _ = mlir.jaxpr_subcomp(
      ctx.module_context, vectorized_jaxpr, name_stack, mlir.TokenSet(),
      const_nodes, *sharded_global_in_nodes, dim_var_values=ctx.dim_var_values)

  sharded_global_out_nodes = [
    mlir.wrap_with_sharding_op(
        ctx, node, aval,
        NamedSharding(mesh, array_mapping_to_axis_resources(aval_axes)
                      )._to_xla_hlo_sharding(aval.ndim).to_proto())
    if aval_axes else node
    for (node,), aval, aval_axes in zip(global_out_nodes, global_out_avals, mesh_out_axes)
  ]

  return sharded_global_out_nodes


def _xmap_lowering_rule_spmd_manual(ctx, *global_in_nodes,
                                    call_jaxpr, name, in_axes, out_axes,
                                    donated_invars, global_axis_sizes, spmd_in_axes,
                                    spmd_out_axes, axis_resources,
                                    resource_env, backend):
  del donated_invars
  assert spmd_in_axes is None and spmd_out_axes is None
  # This first part (up to vtile_manual) is shared with non-MANUAL SPMD rule.
  mlir.check_backend_matches(backend, ctx.module_context.platforms)
  plan = EvaluationPlan.from_axis_resources(
      axis_resources, resource_env, global_axis_sizes)
  manual_mesh_axes = frozenset(it.chain.from_iterable(plan.physical_axis_resources.values()))

  resource_call_jaxpr = plan.subst_axes_with_resources(call_jaxpr)
  f = lu.wrap_init(core.jaxpr_as_fun(core.ClosedJaxpr(resource_call_jaxpr, ())))
  f = hide_mapped_axes(f, in_axes, out_axes)
  f = plan.vectorize_and_loop(f, in_axes, out_axes)

  # NOTE: Sharding constraints are handled entirely by vtile_manual!
  mesh_in_axes, mesh_out_axes = plan.to_mesh_axes(in_axes, out_axes)
  mesh = resource_env.physical_mesh
  f = pxla.vtile_manual(f, tuple(manual_mesh_axes), mesh, mesh_in_axes, mesh_out_axes)

  # NOTE: We don't extend the resource env with the mesh shape, because those
  #       resources are already in scope! It's the outermost xmap that introduces
  #       them!
  global_in_avals = ctx.avals_in
  vectorized_jaxpr, global_out_avals, consts, () = pe.trace_to_jaxpr_dynamic(f, global_in_avals)
  const_nodes = [mlir.ir_constants(xla.canonicalize_dtype(x)) for x in consts]

  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  assert isinstance(ctx.module_context.axis_context,
                    sharding_impls.SPMDAxisContext)
  name_stack = ctx.name_stack.extend(wrap_name(name, 'xmap'))
  sub_ctx = ctx.module_context.replace(
      axis_context=ctx.module_context.axis_context.extend_manual(manual_mesh_axes))
  if any(effects.ordered_effects.contains(eff) for eff
         in vectorized_jaxpr.effects):
    raise NotImplementedError('Cannot lower `xmap` with ordered effects.')
  global_out_nodes, _ = mlir.jaxpr_subcomp(
      sub_ctx, vectorized_jaxpr, name_stack,
      mlir.TokenSet(), const_nodes, *([n] for n in global_in_nodes),
      dim_var_values=ctx.dim_var_values)

  return global_out_nodes


def _tile_base_indices(tile_shape, axes, axis_sizes):
  zero = np.zeros((), dtype=np.int32)
  linear_idxs = [zero] * len(tile_shape)
  strides = [1] * len(tile_shape)
  for name, axis in reversed(axes.items()):
    axis_index = lax.axis_index(name)
    stride_c = np.array(strides[axis], np.int32)
    if linear_idxs[axis] is zero and strides[axis] == 1:
      linear_idxs[axis] = axis_index
    else:
      linear_idxs[axis] = lax.add(linear_idxs[axis],
                                  lax.mul(axis_index, stride_c))
    strides[axis] *= axis_sizes[name]
  return [zero if linear_idx is zero else
          lax.mul(linear_idx, np.array(tile_dim_size, np.int32))
          for linear_idx, tile_dim_size in zip(linear_idxs, tile_shape)]


def _tile(x, in_axes, axis_sizes):
  if not in_axes:
    return x
  tile_shape = list(x.shape)
  for name, axis in in_axes.items():
    axis_size = axis_sizes[name]
    assert tile_shape[axis] % axis_size == 0
    tile_shape[axis] //= axis_size
  base_idxs = _tile_base_indices(tile_shape, in_axes, axis_sizes)
  return lax.dynamic_slice(x, base_idxs, tile_shape)


# TODO(b/110096942): more efficient gather
def _untile(x, out_axes, axis_sizes):
  tile_shape = list(x.shape)
  shape = list(tile_shape)
  for name, axis in out_axes.items():
    shape[axis] *= axis_sizes[name]
  base_idxs = _tile_base_indices(tile_shape, out_axes, axis_sizes)

  padded = lax.broadcast(np.array(0, x.dtype), shape)
  padded = lax.dynamic_update_slice(padded, x, base_idxs)
  out = lax.psum(padded, tuple(out_axes.keys()))
  return out


# -------- helper functions --------

def _delete_aval_axes(aval, axes: AxisNamePos, global_axis_sizes):
  assert isinstance(aval, core.ShapedArray)
  shape = list(aval.shape)
  named_shape = dict(aval.named_shape)
  for name, dim in sorted(axes.items(), key=lambda x: x[1], reverse=True):
    named_shape[name] = global_axis_sizes[name]
    del shape[dim]
  return aval.update(shape=tuple(shape), named_shape=named_shape)

def _insert_aval_axes(aval, axes: AxisNamePos, local_axis_sizes):
  assert isinstance(aval, core.ShapedArray)
  shape = list(aval.shape)
  named_shape = dict(aval.named_shape)
  for name, dim in sorted(axes.items(), key=lambda x: x[1]):
    shape.insert(dim, local_axis_sizes[name])
    named_shape.pop(name, None)  # The name might be missing --- it's a broadcast.
  return aval.update(shape=tuple(shape), named_shape=named_shape)


class ResourceCount(NamedTuple):
  nglobal: int
  nlocal: int | None
  distributed: bool

  def to_local(self, global_size):
    return global_size


def _get_axis_resource_count(
    axis_resources, resource_env) -> dict[ResourceAxisName, ResourceCount]:
  global_res_shape = resource_env.shape
  local_res_shape = None

  distributed = (False if resource_env.physical_mesh.empty else
                 resource_env.physical_mesh.size != len(resource_env.physical_mesh.local_devices))
  resource_count_map = {}
  for axis, resources in axis_resources.items():
    if local_res_shape is None:
      nlocal = None
    else:
      nlocal = math.prod(map(local_res_shape.get, resources))
    resource_count_map[axis] = ResourceCount(
        math.prod(map(global_res_shape.get, resources)),
        nlocal, distributed)
  return resource_count_map


def _get_axis_sizes(args_flat: Iterable[Any],
                    in_axes_flat: Iterable[AxisNamePos],
                    global_axis_sizes: dict[AxisName, int],
                    axis_resource_count: dict[AxisName, ResourceCount]):
  global_axis_sizes = dict(global_axis_sizes)
  for arg, in_axes in zip(args_flat, in_axes_flat):
    for name, dim in in_axes.items():
      try:
        dim_size = arg.shape[dim]
      except IndexError:
        # TODO(apaszke): Handle negative indices. Check for overlap too!
        raise ValueError(f"One of xmap arguments has an in_axes specification of "
                         f"{in_axes.user_repr}, which implies that it has at least "
                         f"{max(in_axes.values()) + 1} dimensions, but the argument "
                         f"has rank {arg.ndim}")
      global_dim_size = dim_size
      if name in global_axis_sizes:
        expected_global_dim_size = global_axis_sizes[name]
        if global_dim_size != expected_global_dim_size:
          raise ValueError(f"The size of axis {name} was previously inferred to be "
                            f"{expected_global_dim_size}, but found an argument of shape {arg.shape} "
                            f"with in_axes specification {in_axes.user_repr}. Shape mismatch "
                            f"occurs in dimension {dim}: {global_dim_size} != {expected_global_dim_size}")
      global_axis_sizes[name] = global_dim_size
  return FrozenDict(global_axis_sizes)


@lu.transformation
def hide_mapped_axes(flat_in_axes, flat_out_axes, *flat_args):
  def _squeeze_mapped_axes(arg, axes: AxisNamePos):
    for dim in sorted(axes.values(), reverse=True):
      arg = arg.squeeze(dim)
    return arg

  def _unsqueeze_mapped_axes(out, axes: AxisNamePos):
    try:
      return jnp.expand_dims(out, tuple(axes.values()))
    except ValueError as e:
      # Improve the axis out of bounds errors
      # TODO(apaszke): Handle negative indices. Check for overlap too!
      if e.args[0].startswith('axis') and 'out of bounds' in e.args[0]:
        raise ValueError(f"One of xmap outputs has an out_axes specification of "
                         f"{axes.user_repr}, which requires the result of the xmapped "
                         f"function to have at least {max(axes.values()) - len(axes) + 1} "
                         f"positional dimensions, but it only has {out.ndim}")
      raise

  squeezed_args = map(_squeeze_mapped_axes, flat_args, flat_in_axes)
  flat_outputs = yield squeezed_args, {}
  yield map(_unsqueeze_mapped_axes, flat_outputs, flat_out_axes)


def _jaxpr_resources(jaxpr, resource_env) -> set[ResourceAxisName]:
  if isinstance(jaxpr, core.ClosedJaxpr):
    jaxpr = jaxpr.jaxpr
  assert isinstance(jaxpr, core.Jaxpr)
  used_resources = set()
  for eqn in jaxpr.eqns:
    if eqn.primitive is xmap_p:
      if eqn.params['resource_env'].physical_mesh != resource_env.physical_mesh:
        raise RuntimeError("Changing the physical mesh is not allowed inside xmap.")
      used_resources |= set(it.chain(*eqn.params['axis_resources'].values()))
    updates = core.traverse_jaxpr_params(
        partial(_jaxpr_resources, resource_env=resource_env), eqn.params).values()
    for update in updates:
      used_resources |= update
  return used_resources


def _to_resource_axes(axes_specs: Sequence[AxisNamePos],
                      axis_resources: dict[AxisName, tuple[ResourceAxisName, ...]]):
  """
  Convert in/out_axes parameters ranging over logical dimensions to
  ones that range over resource dimensions.

  Note that values no longer have to be distinct, as multiple resource
  axes can tile a single positional axes. This is why the result is
  an OrderedDict with an implicit major-to-minor ordering.
  """
  return tuple(OrderedDict((resource_axis, pos_axis)
                           for logical_axis, pos_axis in axes.items()
                           for resource_axis in axis_resources[logical_axis])
               for axes in axes_specs)


def _merge_leading_axis(x, axis: int | None):
  if axis is None:
    # We assume that the output does not vary along the leading axis
    return lax.index_in_dim(x, 0, axis=0, keepdims=False)
  else:
    x_moved = moveaxis(x, 0, axis)
    shape = list(x_moved.shape)
    shape[axis:axis + 2] = [shape[axis] * shape[axis + 1]]
    return x_moved.reshape(shape)


def _slice_tile(x, dim: int | None, i, n: int):
  """Selects an `i`th (out of `n`) tiles of `x` along `dim`."""
  if dim is None: return x
  (tile_size, rem) = divmod(x.shape[dim], n)
  assert rem == 0, "Please open a bug report!"
  return lax.dynamic_slice_in_dim(x, i * tile_size, slice_size=tile_size, axis=dim)


def _unzip_axis_resources(axis_resources: dict[AxisName, tuple[ResourceAxisName, ...]],
                          resource_env: ResourceEnv):
  """Splits axis_resources into separate dicts for physical and loop resources."""
  physical_axis_resources = {}
  loop_axis_resources = {}
  loop_resource_axes = resource_env.loop_resource_axes
  for axis, raxes in axis_resources.items():
    first_loop = 0
    for raxis in raxes:
      if raxis in loop_resource_axes:
        break
      else:
        first_loop += 1
    physical_axis_resources[axis] = raxes[:first_loop]
    loop_resources = loop_axis_resources[axis] = raxes[first_loop:]
    if not all(name in loop_resource_axes for name in loop_resources):
      raise NotImplementedError("Loop resources cannot appear before mesh axes "
                                "in the resource_axis argument")
  return physical_axis_resources, loop_axis_resources


def _check_out_avals_vs_out_axes(out_avals: Sequence[core.AbstractValue],
                                 out_axes: Sequence[AxisNamePos],
                                 global_axis_sizes: dict[AxisName, int]):
  defined_axes = set(global_axis_sizes)
  for aval, axes in zip(out_avals, out_axes):
    if not isinstance(aval, core.ShapedArray):
      if axes:
        raise AssertionError(f"Only array abstract values can have non-empty "
                             f"out_axes, but {aval} has {axes}")
      continue
    undeclared_axes = (set(aval.named_shape) - set(axes)) & defined_axes
    if undeclared_axes:
      undeclared_axes_str = sorted(str(axis) for axis in undeclared_axes)
      raise TypeError(f"One of xmap results has an out_axes specification of "
                      f"{axes.user_repr}, but is actually mapped along more axes "
                      f"defined by this xmap call: {', '.join(undeclared_axes_str)}")


def _check_gda_or_array_xmap_partitioning(axis_resources, resource_env,
                                          global_axis_sizes, in_axes_flat,
                                          args_flat):
  @lru_cache
  def _check_sharding(in_sharding, xmap_sharding, ndim, arr_flavor):
    if (not op_shardings.are_op_shardings_equal(
          in_sharding._to_xla_hlo_sharding(ndim),
          xmap_sharding._to_xla_hlo_sharding(ndim)) or
        in_sharding.memory_kind != xmap_sharding.memory_kind):
      raise ValueError(
          f"Got an input {arr_flavor} to xmap with different partitioning than "
          "specified in xmap. The partitioning must match. "
          f"Got {arr_flavor} spec: {in_sharding.spec} and "
          f"xmap spec: {xmap_sharding.spec}")

  mesh_in_axes = EvaluationPlan.from_axis_resources(  # pytype: disable=wrong-arg-types  # always-use-return-annotations
      axis_resources, resource_env, global_axis_sizes).to_mesh_axes(in_axes_flat)
  for arg, xmap_array_mapping in safe_zip(args_flat, mesh_in_axes):
    if isinstance(arg, ArrayImpl):
      if not isinstance(arg.sharding, NamedSharding):
        continue
      mesh = arg.sharding.mesh
      if mesh != resource_env.physical_mesh:
        raise ValueError("xmap's mesh and Array's mesh should be equal. "
                         f"Got xmap mesh: {resource_env.physical_mesh},\n"
                         f"Array mesh: {mesh}")

      s = arg.sharding
      xmap_sharding = pxla.create_mesh_pspec_sharding(
          mesh, array_mapping_to_axis_resources(xmap_array_mapping))
      # This check is cached because comparing OpSharding is expensive during
      # dispatch and if the shardings are the same, then there is no need to
      # compare twice.
      _check_sharding(s, xmap_sharding, arg.ndim, 'Array')


# TODO: We should relax this at least for "constructor primitives"
#       such as axis_index or zeros.
def _check_no_loop_collectives(jaxpr, loop_axis_resources):
  if isinstance(jaxpr, core.ClosedJaxpr):
    jaxpr = jaxpr.jaxpr
  def subst_no_loop(name):
    if loop_axis_resources.get(name, ()):
      raise RuntimeError(f"Named axes with loop resources assigned to them cannot "
                          f"be referenced inside the xmapped computation (e.g. in "
                          f"collectives), but `{name}` violates that rule")
    return (name,)
  for eqn in jaxpr.eqns:
    core.subst_axis_names(eqn.primitive, eqn.params, subst_no_loop, traverse=False)
    rec = partial(_check_no_loop_collectives, loop_axis_resources=loop_axis_resources)
    core.traverse_jaxpr_params(rec, eqn.params)


def _fix_inferred_spmd_sharding(jaxpr, resource_env, gen_fresh_name = None):
  rec = lambda jaxpr: _fix_inferred_spmd_sharding(jaxpr, resource_env, gen_fresh_name)
  if isinstance(jaxpr, core.ClosedJaxpr):
    return jaxpr.map_jaxpr(rec)
  assert isinstance(jaxpr, core.Jaxpr)
  if gen_fresh_name is None:
    gen_fresh_name = core.gensym()
  new_eqns = []
  for eqn in jaxpr.eqns:
    new_jaxpr_params = core.traverse_jaxpr_params(rec, eqn.params)
    tmp_outvars = [gen_fresh_name(v.aval) for v in eqn.outvars]
    new_eqns.append(eqn.replace(
      outvars=tmp_outvars, params=dict(eqn.params, **new_jaxpr_params)))
    for outvar, tmpvar in zip(eqn.outvars, tmp_outvars):
      mps = NamedSharding._from_parsed_pspec(
          resource_env.physical_mesh, ParsedPartitionSpec((), ()))
      unconstrained_dims = get_unconstrained_dims(mps)
      gspmd_sharding = GSPMDSharding.get_replicated(mps._device_assignment)
      new_eqns.append(core.JaxprEqn(
          [tmpvar], [outvar], sharding_constraint_p,
          dict(resource_env=resource_env,
               sharding=gspmd_sharding,
               unconstrained_dims=unconstrained_dims),
          set(),
          eqn.source_info))
  return jaxpr.replace(eqns=new_eqns)

def _flatten_axes(what, tree, axes, tupled_args):
  try:
    return tuple(flatten_axes(what, tree, axes, tupled_args=tupled_args))
  except ValueError:
    pass
  # Replace axis_resources with unparsed versions to avoid revealing internal details
  flatten_axes(what, tree, tree_map(lambda parsed: NoQuotesStr(parsed.user_repr), axes),
               tupled_args=tupled_args)
  raise AssertionError("Please open a bug request!")  # This should be unreachable

class NoQuotesStr(str):
  __repr__ = str.__str__


# -------- config flags --------

def _thread_local_flag_unsupported(_):
  raise RuntimeError("thread-local xmap flags not supported!")
def _clear_compilation_cache(_):
  make_xmap_callable.cache_clear()  # type: ignore

def _ensure_spmd_and(f):
  def update(v):
    if v and not SPMD_LOWERING.value:
      raise RuntimeError("This flag requires enabling the experimental_xmap_spmd_lowering flag")
    return f(v)
  return update


SPMD_LOWERING = config.define_bool_state(
    name="experimental_xmap_spmd_lowering",
    default=False,
    help=("When set, multi-device xmap computations will be compiled through "
          "the XLA SPMD partitioner instead of explicit cross-replica collectives. "
          "Not supported on CPU!"),
    update_global_hook=_clear_compilation_cache,
    update_thread_local_hook=_thread_local_flag_unsupported)
SPMD_LOWERING_MANUAL = config.define_bool_state(
    name="experimental_xmap_spmd_lowering_manual",
    default=False,
    help=("When set, multi-device xmap computations will be compiled using "
          "the MANUAL partitioning feature of the XLA SPMD partitioner instead of "
          "sharding constraints on vectorized code. "
          "Requires experimental_xmap_spmd_lowering!"),
    update_global_hook=_ensure_spmd_and(_clear_compilation_cache),
    update_thread_local_hook=_thread_local_flag_unsupported)
_ENSURE_FIXED_SHARDING = config.define_bool_state(
    name="experimental_xmap_ensure_fixed_sharding",
    default=False,
    help=("When set and `experimental_xmap_spmd_lowering` is enabled, the lowering will "
          "try to limit the flexibility of the automated SPMD partitioner heuristics "
          "by emitting additional sharding annotations for program intermediates."),
    update_global_hook=_ensure_spmd_and(_clear_compilation_cache),
    update_thread_local_hook=_thread_local_flag_unsupported)
