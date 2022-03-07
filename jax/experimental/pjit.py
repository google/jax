# Copyright 2021 Google LLC
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

from enum import IntEnum
import numpy as np
from collections import OrderedDict, Counter
from typing import Callable, Sequence, Tuple, Union, Optional
from warnings import warn
import itertools as it
from functools import partial

from jax.experimental import maps
from jax.experimental.global_device_array import GlobalDeviceArray as GDA
from jax import core
from jax import linear_util as lu
from jax._src.api import _check_callable, _check_arg, Lowered
from jax._src import dispatch
from jax._src import source_info_util
from jax._src.api_util import (argnums_partial_except, flatten_axes,
                               flatten_fun_nokwargs, _ensure_index_tuple,
                               donation_vector, rebase_donate_argnums,
                               shaped_abstractify)
from jax.errors import JAXTypeError
from jax.interpreters import ad
from jax.interpreters import mlir
from jax.interpreters import pxla
from jax.interpreters import xla
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters.sharded_jit import PartitionSpec
from jax._src.lib import xla_client as xc
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           treedef_is_leaf, tree_structure, treedef_tuple)
from jax._src.tree_util import prefix_errors
from jax._src.util import (extend_name_stack, HashableFunction, safe_zip,
                         wrap_name, wraps, distributed_debug_log,
                         split_list, cache, tuple_insert)
xops = xc._xla.ops

class _FromGdaSingleton:
  pass
FROM_GDA = _FromGdaSingleton()

def _is_from_gda(x):
  # It's occasionally possible to end up with two FROM_GDA singletons (e.g. if
  # pickling in_axis_resources and sending to other processes). Make sure this
  # doesn't cause an error to avoid user confusion.
  return isinstance(x, type(FROM_GDA))


# TODO(yashkatariya): Add pjit microbenchmarks.
def pjit(fun: Callable,
         in_axis_resources,
         out_axis_resources,
         static_argnums: Union[int, Sequence[int]] = (),
         donate_argnums: Union[int, Sequence[int]] = ()):
  """Makes ``fun`` compiled and automatically partitioned across multiple devices.

  The returned function has semantics equivalent to those of ``fun``, but is
  compiled to an XLA computation that runs across multiple devices
  (e.g. multiple GPUs or multiple TPU cores). This can be useful if the jitted
  version of ``fun`` would not fit in a single device's memory, or to speed up
  ``fun`` by running each operation in parallel across multiple devices.

  The partitioning over devices happens automatically based on the
  propagation of the input partitioning specified in ``in_axis_resources`` and
  the output partitioning specified in ``out_axis_resources``. The resources
  specified in those two arguments must refer to mesh axes, as defined by
  the :py:func:`jax.experimental.maps.Mesh` context manager. Note that the mesh
  definition at ``pjit`` application time is ignored, and the returned function
  will use the mesh definition available at each call site.

  Inputs to a pjit'd function will be automatically partitioned across devices
  if they're not already correctly partitioned based on ``in_axis_resources``.
  In some scenarios, ensuring that the inputs are already correctly pre-partitioned
  can increase performance. For example, if passing the output of one pjit'd function
  to another pjit’d function (or the same pjit’d function in a loop), make sure the
  relevant ``out_axis_resources`` match the corresponding ``in_axis_resources``.

  .. note::
    **Multi-process platforms:** On multi-process platforms such as TPU pods,
    ``pjit`` can be used to run computations across all available devices across
    processes. To achieve this, ``pjit`` is designed to be used in SPMD Python
    programs, where every process is running the same Python code such that all
    processes run the same pjit'd function in the same order.

    When running in this configuration, the mesh should contain devices across
    all processes. However, any input argument dimensions partitioned over
    multi-process mesh axes should be of size equal to the corresponding *local*
    mesh axis size, and outputs will be similarly sized according to the local
    mesh. ``fun`` will still be executed across *all* devices in the mesh,
    including those from other processes, and will be given a global view of the
    data spread across multiple processes as a single array. However, outside
    of ``pjit`` every process only "sees" its local piece of the input and output,
    corresponding to its local sub-mesh.

    This means that each process's participating local devices must form a
    _contiguous_ local sub-mesh within the full global mesh. A contiguous
    sub-mesh is one where all of its devices are adjacent within the global
    mesh, and form a rectangular prism.

    The SPMD model also requires that the same multi-process ``pjit``'d
    functions must be run in the same order on all processes, but they can be
    interspersed with arbitrary operations running in a single process.

  Args:
    fun: Function to be compiled. Should be a pure function, as side-effects may
      only be executed once. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.
    in_axis_resources: Pytree of structure matching that of arguments to ``fun``,
      with all actual arguments replaced by resource assignment specifications.
      It is also valid to specify a pytree prefix (e.g. one value in place of a
      whole subtree), in which case the leaves get broadcast to all values in
      that subtree.

      The valid resource assignment specifications are:
        - :py:obj:`None`, in which case the value will be replicated on all devices
        - :py:class:`PartitionSpec`, a tuple of length at most equal to the rank
          of the partitioned value. Each element can be a :py:obj:`None`, a mesh
          axis or a tuple of mesh axes, and specifies the set of resources assigned
          to partition the value's dimension matching its position in the spec.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it.
    out_axis_resources: Like ``in_axis_resources``, but specifies resource
      assignment for function outputs.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If ``static_argnums`` is not provided, no arguments are treated as static.
    donate_argnums: Specify which argument buffers are "donated" to the computation.
      It is safe to donate argument buffers if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to.

      For more details on buffer donation see the [FAQ](https://jax.readthedocs.io/en/latest/faq.html#buffer-donation).
  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation and
    automaticly partitioned by the mesh available at each call site.

  For example, a convolution operator can be automatically partitioned over
  an arbitrary set of devices by a single ```pjit`` application:

  >>> import jax
  >>> import jax.numpy as jnp
  >>> from jax.experimental.maps import Mesh
  >>> from jax.experimental.pjit import PartitionSpec, pjit
  >>>
  >>> x = jnp.arange(8, dtype=jnp.float32)
  >>> f = pjit(lambda x: jax.numpy.convolve(x, jnp.asarray([0.5, 1.0, 0.5]), 'same'),
  ...         in_axis_resources=None, out_axis_resources=PartitionSpec('devices'))
  >>> with Mesh(jax.devices(), ('devices',)):
  ...   print(f(x))  # doctest: +SKIP
  [ 0.5  2.   4.   6.   8.  10.  12.  10. ]
  """
  warn("pjit is an experimental feature and probably has bugs!")
  _check_callable(fun)

  if isinstance(in_axis_resources, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/google/jax/issues/2367
    in_axis_resources = tuple(in_axis_resources)

  in_axis_resources, _, _ = _prepare_axis_resources(
      in_axis_resources, "in_axis_resources")
  out_axis_resources, _, _ = _prepare_axis_resources(
      out_axis_resources, "out_axis_resources")

  static_argnums = _ensure_index_tuple(static_argnums)
  donate_argnums = _ensure_index_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  def infer_params(*args, **kwargs):
    if kwargs:
      raise NotImplementedError("pjit does not support kwargs")
    if max(static_argnums + donate_argnums, default=-1) >= len(args):
      raise ValueError(f"jitted function has static_argnums={static_argnums}, "
                       f"donate_argnums={donate_argnums} but "
                       f"was called with only {len(args)} positional arguments.")

    # Putting this outside of wrapped would make resources lexically scoped
    resource_env = pxla.thread_resources.env
    mesh = resource_env.physical_mesh
    if mesh.empty:
      raise RuntimeError("pjit requires a non-empty mesh! Are you sure that "
                         "it's defined at the call site?")
    if any(d.platform not in {'gpu', 'tpu'} for d in mesh.devices.flat):
      raise RuntimeError("pjit only supports GPU and TPU devices")

    f = lu.wrap_init(fun)
    f, dyn_args = argnums_partial_except(f, static_argnums, args, allow_invalid=False)
    del args

    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    if donate_argnums:
      donated_invars = donation_vector(donate_argnums, dyn_args, ())
    else:
      donated_invars = (False,) * len(args_flat)

    _maybe_check_pjit_gda_mesh(args_flat, mesh)

    local_in_avals = tuple(shaped_abstractify(a) for a in args_flat)
    # TODO(yashkatariya): This is a hack. This should go away when avals have
    # is_global attribute.
    in_positional_semantics = tuple(
        maps._PositionalSemantics.GLOBAL if isinstance(a, GDA) else maps._positional_semantics.val
        for a in args_flat)
    out_positional_semantics = maps._positional_semantics.val
    jaxpr, in_axis_resources_flat, out_axis_resources_flat = _pjit_jaxpr(
        flat_fun, mesh, local_in_avals, in_tree,
        hashable_pytree(in_axis_resources),
        HashableFunction(out_tree, closure=()),
        hashable_pytree(out_axis_resources),
        in_positional_semantics, out_positional_semantics,
        tuple(isinstance(a, GDA) for a in args_flat))
    in_axis_resources_flat = tree_map(_maybe_replace_from_gda_with_pspec,
                                      in_axis_resources_flat, tuple(args_flat))
    params = dict(
        jaxpr=jaxpr,
        in_axis_resources=in_axis_resources_flat,
        out_axis_resources=out_axis_resources_flat,
        resource_env=resource_env,
        donated_invars=donated_invars,
        name=getattr(flat_fun, '__name__', '<unnamed function>'),
        in_positional_semantics=in_positional_semantics,
        out_positional_semantics=out_positional_semantics)
    return (args_flat, local_in_avals, params, in_tree, out_tree(),
            donate_argnums)

  @wraps(fun)
  def wrapped(*args, **kwargs):
    args_flat, _, params, _, out_tree, _ = infer_params(*args, **kwargs)
    for arg in args_flat:
      _check_arg(arg)
    out = pjit_p.bind(*args_flat, **params)
    return tree_unflatten(out_tree, out)

  def lower(*args, **kwargs):
    (args_flat, flat_local_in_avals, params, in_tree, out_tree,
     donate_argnums) = infer_params(*args, **kwargs)
    lowering = _pjit_lower(
        params['jaxpr'], params['in_axis_resources'],
        params['out_axis_resources'], params['resource_env'],
        params['donated_invars'], params['name'],
        params['in_positional_semantics'], params['out_positional_semantics'])

    args_kwargs_in_tree = treedef_tuple([in_tree, tree_flatten({})[1]])
    local_in_avals = args_kwargs_in_tree.unflatten(flat_local_in_avals)
    return Lowered(lowering, args_kwargs_in_tree, local_in_avals, out_tree,
                   donate_argnums, no_kwargs=True)

  wrapped.lower = lower
  return wrapped

class _ListWithW(list):
  __slots__ = ('__weakref__',)

def hashable_pytree(pytree):
  vals, treedef = tree_flatten(pytree)
  vals = tuple(vals)
  return HashableFunction(lambda: tree_unflatten(treedef, vals),
                          closure=(treedef, vals))

def flatten_axis_resources(what, tree, axis_resources, tupled_args):
  try:
    return tuple(flatten_axes(what, tree, axis_resources, tupled_args=tupled_args))
  except ValueError:
    pass  # Raise a tree prefix error below

  # Tree leaves are always valid prefixes, so if there was a prefix error as
  # assumed here, axis_resources must not be a leaf.
  assert not treedef_is_leaf(tree_structure(axis_resources))

  # Check the type directly rather than using isinstance because of namedtuples.
  if tupled_args and (type(axis_resources) is not tuple or
                      len(axis_resources) != len(tree.children())):
    # We know axis_resources is meant to be a tuple corresponding to the args
    # tuple, but while it is a non-leaf pytree, either it wasn't a tuple or it
    # wasn't the right length.
    msg = (f"{what} specification must be a tree prefix of the positional "
           f"arguments tuple passed to the `pjit`-decorated function. In "
           f"particular, {what} must either be a None, a PartitionSpec, or "
           f"a tuple of length equal to the number of positional arguments.")
    # If `tree` represents an args tuple, then `axis_resources` must be a tuple.
    # TODO(mattjj,apaszke): disable implicit list casts, remove 'or list' below
    if type(axis_resources) is not tuple:
      msg += f" But {what} is not a tuple: got {type(axis_resources)} instead."
    elif len(axis_resources) != len(tree.children()):
      msg += (f" But {what} is the wrong length: got a tuple or list of length "
              f"{len(axis_resources)} for an args tuple of length "
              f"{len(tree.children())}.")

    # As an extra hint, let's check if the user just forgot to wrap
    # in_axis_resources in a singleton tuple.
    if len(tree.children()) == 1:
      try: flatten_axes(what, tree, (axis_resources,))
      except ValueError: pass  # That's not the issue.
      else:
        msg += (f" Given the corresponding argument being "
                f"passed, it looks like {what} might need to be wrapped in "
                f"a singleton tuple.")

    raise ValueError(msg)

  # Replace axis_resources with unparsed versions to avoid revealing internal details
  axis_tree = tree_map(lambda parsed: parsed.user_spec, axis_resources)

  # Because ecause we only have the `tree` treedef and not the full pytree here,
  # we construct a dummy tree to compare against. Revise this in callers?
  dummy_tree = tree_unflatten(tree, [PytreeLeaf()] * tree.num_leaves)
  errors = prefix_errors(axis_tree, dummy_tree)
  if errors:
    e = errors[0]  # Only show information about the first disagreement found.
    raise e(what)

  # At this point we've failed to find a tree prefix error.
  assert False, "Please open a bug report!"  # This should be unreachable.

class PytreeLeaf:
  def __repr__(self): return "pytree leaf"


@lu.cache
def _pjit_jaxpr(fun, mesh, local_in_avals,
                in_tree, in_axis_resources_thunk,
                out_tree, out_axis_resources_thunk,
                in_positional_semantics, out_positional_semantics, is_gda):
  # TODO(yashkatariya): Make this work with FROM_GDA special value.
  in_axis_resources_flat = flatten_axis_resources(
        "pjit in_axis_resources", in_tree,
        in_axis_resources_thunk(), tupled_args=True)
  canonicalized_in_axis_resources_flat = tree_map(_create_cpspec, in_axis_resources_flat)
  # This check should be above local_to_global call below otherwise if
  # `FROM_GDA` is passed to any input other than GDA, a ugly error message
  # will be raised because get_array_mapping (in local_to_global) of a
  # FROM_GDA cannot happen.
  tree_map(_check_resources_mismatch, in_axis_resources_flat, is_gda)
  # If all inputs are either GDAs or fully replicated, then the avals are
  # global and the mesh should also be global. This split is because
  # non-contiguous mesh can only be used if all inputs are either GDAs or fully
  # replicated.
  # Use canonicalized in_axis_resources here because we want to treat P(None)
  # and None (for example) as equivalent.
  if all(((not _is_from_gda(p) and p.partitions == ()) or ig)
         for p, ig in safe_zip(canonicalized_in_axis_resources_flat, is_gda)):
    # Shapes should be checked against non canonicalized in_axis_resources.
    # For example, partitions of () and ((),) are not equivalent, since the
    # first one is a valid spec for a scalar value, while the second is not!
    _check_shapes_against_resources(
        "pjit arguments", mesh.is_multi_process, mesh.shape, local_in_avals,
        in_axis_resources_flat, allow_uneven_sharding=False)
  else:
    _check_shapes_against_resources("pjit arguments", False, mesh.local_mesh.shape,
                                    local_in_avals, in_axis_resources_flat,
                                    allow_uneven_sharding=False)

  global_in_avals = local_to_global(in_positional_semantics, mesh,
                                    local_in_avals, canonicalized_in_axis_resources_flat)

  prev_positional_val = maps._positional_semantics.val
  try:
    maps._positional_semantics.val = maps._PositionalSemantics.GLOBAL
    with dispatch.log_elapsed_time(f"Finished tracing + transforming {fun.__name__} "
                                   "for pjit in {elapsed_time} sec"):
      jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, global_in_avals)
  finally:
    maps._positional_semantics.val = prev_positional_val
  jaxpr = core.ClosedJaxpr(jaxpr, consts)

  out_axis_resources_flat = flatten_axis_resources(
        "pjit out_axis_resources", out_tree(),
        out_axis_resources_thunk(), tupled_args=False)
  _check_shapes_against_resources("pjit outputs", mesh.is_multi_process, mesh.shape,
                                  global_out_avals, out_axis_resources_flat,
                                  allow_uneven_sharding=False)
  canonicalized_out_axis_resources_flat = tree_map(_create_cpspec, out_axis_resources_flat)
  # lu.cache needs to be able to create weakrefs to outputs, so we can't return a plain tuple
  return _ListWithW([jaxpr, canonicalized_in_axis_resources_flat,
                     canonicalized_out_axis_resources_flat])


class SpecSync(IntEnum):
  """Encodes how much out of sync the real value of partitions is compared to the user specified one.

  We use this to make sure we don't show garbage modified values while claiming
  that the users have specified them like that.
  """
  OUT_OF_SYNC = 0  # Arbitrary changes, including new axes inserted
  DIM_PERMUTE = 1  # Dimensions permuted, but no new sharding axes
  IN_SYNC = 2  # Entirely in sync

class ParsedPartitionSpec:
  __slots__ = ('unsafe_user_spec', 'partitions', 'sync')

  def __init__(self, user_spec, partitions, sync=SpecSync.IN_SYNC):
    self.unsafe_user_spec = user_spec
    # None in partitions represents unconstrained dim.
    # TODO(yashkatariya): May use a sentinel value.
    self.partitions = tuple(partitions)
    self.sync = sync

  @property
  def user_spec(self):
    return self.unsynced_user_spec(SpecSync.IN_SYNC)

  def unsynced_user_spec(self, min_sync):
    if self.sync < min_sync:
      raise AssertionError(f"Please open a bug report! ({self.sync} >= {min_sync})")
    return self.unsafe_user_spec

  def insert_axis_partitions(self, dim, val):
    parts = self.partitions
    too_short = dim - len(parts)
    if too_short > 0:
      parts += ((),) * too_short
    new_partitions = tuple_insert(parts, dim, val)
    new_sync = SpecSync.DIM_PERMUTE if (val == () or val is None) else SpecSync.OUT_OF_SYNC
    return ParsedPartitionSpec(self.unsafe_user_spec, new_partitions, sync=new_sync)

  @classmethod
  def from_user_input(cls, entry, arg_name, allow_unconstrained_dims=False):
    if entry is None:
      return cls(entry, ())
    if not isinstance(entry, PartitionSpec):
      raise TypeError(f"{arg_name} are expected to be "
                      f"PartitionSpec instances or None, but got {entry}")
    axis_specs = []
    for axis_spec in entry:
      if axis_spec is None:
        axis_spec = ()
      elif isinstance(axis_spec, (list, tuple)):
        axis_spec = tuple(axis_spec)
      elif axis_spec == PartitionSpec.UNCONSTRAINED:
        if not allow_unconstrained_dims:
          raise ValueError(f"Unconstrained dims are not allowed: {entry}")
        axis_spec = None
      else:
        axis_spec = (axis_spec,)
      axis_specs.append(axis_spec)
    return cls(entry, axis_specs)

  def __hash__(self):
    return hash((self.partitions, self.sync))

  def __eq__(self, other):
    return (self.partitions == other.partitions and
            self.sync == other.sync)

  def __len__(self):
    return len(self.partitions)

  def __getitem__(self, i):
    return self.partitions[i]

  def __iter__(self):
    return iter(self.partitions)

  def __repr__(self):
    return (f"ParsedPartitionSpec(partitions={self.partitions}, "
            f"unsafe_user_spec={self.unsafe_user_spec}, "
            f"sync={self.sync})")

REPLICATED = ParsedPartitionSpec(None, ())


class CanonicalizedParsedPartitionSpec(ParsedPartitionSpec):
  """ParsedPartitionSpecs that are canonicalized.

  ParsedPartitionSpecs may contain trailing empty tuples, that make them
  semantically different in general, and yet in some situations we prefer
  to regard them as equivalent. For example, partitions of () and ((),)
  cannot be always considered equivalent, since the first one is a valid
  spec for a scalar value, while the second is not! However, when either of
  those are applied to a 2D array, they both mean that the array is fully
  replicated.

  So CanonicalizedParsedPartitionSpecs removes the trailing empty tuples from
  partitions.
  """

  def __init__(self, parsed_pspec: ParsedPartitionSpec):
    partitions = list(parsed_pspec.partitions)
    while partitions and partitions[-1] == ():
      partitions.pop()

    super().__init__(parsed_pspec.unsafe_user_spec, partitions,
                     parsed_pspec.sync)

  def __repr__(self):
    return (f"CanonicalizedParsedPartitionSpec(partitions={self.partitions}, "
            f"unsafe_user_spec={self.unsafe_user_spec}, "
            f"sync={self.sync})")


def _prepare_axis_resources(axis_resources,
                            arg_name,
                            allow_unconstrained_dims=False):
  # PyTrees don't treat None values as leaves, so we use an is_leaf function.
  entries, treedef = tree_flatten(axis_resources, is_leaf=lambda x: x is None)
  what = f"{arg_name} leaf specifications"
  entries = [
      entry if _is_from_gda(entry) else ParsedPartitionSpec.from_user_input(
          entry, what, allow_unconstrained_dims=allow_unconstrained_dims)
      for entry in entries
  ]
  _check_unique_resources(entries, arg_name)
  return tree_unflatten(treedef, entries), entries, treedef

def _check_resources_mismatch(in_axis_resources_flat, is_gda):
  if not is_gda and _is_from_gda(in_axis_resources_flat):
    raise ValueError('For a non-GDA input, the corresponding resource in '
                     'in_axis_resources cannot be `pjit.FROM_GDA`.')

def _check_unique_resources(axis_resources, arg_name):
  for arg_axis_resources in axis_resources:
    if not arg_axis_resources: continue
    if _is_from_gda(arg_axis_resources): continue
    constrained_dims = [d for d in arg_axis_resources if d is not None]
    resource_counts = Counter(it.chain.from_iterable(constrained_dims))
    if not resource_counts: continue
    if resource_counts.most_common(1)[0][1] > 1:
      multiple_uses = [r for r, c in resource_counts.items() if c > 1]
      if multiple_uses:
        raise ValueError(f"A single {arg_name} specification can map every mesh axis "
                         f"to at most one positional dimension, but {arg_axis_resources.user_spec} "
                         f"has duplicate entries for {pxla.show_axes(multiple_uses)}")

def _check_shapes_against_resources(what: str, is_global_shape: bool,
                                    mesh_shape, flat_avals, flat_axis_resources,
                                    allow_uneven_sharding: bool):
  global_str = " global" if is_global_shape else ""
  for aval, aval_axis_resources in zip(flat_avals, flat_axis_resources):
    if _is_from_gda(aval_axis_resources):
      continue
    shape = aval.shape
    if len(shape) < len(aval_axis_resources):
      raise ValueError(f"One of {what} was given the resource assignment "
                       f"of {aval_axis_resources.user_spec}, which implies that "
                       f"it has a rank of at least {len(aval_axis_resources)}, "
                       f"but it is {len(shape)}")
    for i, axis_resources in enumerate(aval_axis_resources):
      if axis_resources is None:
        continue
      try:
        size = int(np.prod([mesh_shape[resource] for resource in axis_resources], dtype=np.int64))
      except KeyError as e:
        raise ValueError(f"One of {what} was given the resource assignment "
                         f"of {aval_axis_resources.user_spec}, but resource axis "
                         f"{e.args[0]} is undefined. Did you forget to declare the mesh?") from None
      if not allow_uneven_sharding and shape[i] % size != 0:
        raise ValueError(f"One of {what} was given the resource assignment "
                         f"of {aval_axis_resources.user_spec}, which implies that "
                         f"the{global_str} size of its dimension {i} should be "
                         f"divisible by {size}, but it is equal to {shape[i]}")

# -------------------- pjit rules --------------------

pjit_p = core.Primitive("pjit")
pjit_p.multiple_results = True


def _pjit_call_impl(*args, jaxpr,
                    in_axis_resources, out_axis_resources,
                    resource_env, donated_invars, name,
                    in_positional_semantics, out_positional_semantics):
  compiled = _pjit_lower(
      jaxpr, in_axis_resources, out_axis_resources,
      resource_env, donated_invars, name, in_positional_semantics,
      out_positional_semantics).compile()
  distributed_debug_log(("Running pjit'd function", name),
                        ("mesh", resource_env.physical_mesh))
  return compiled.unsafe_call(*args)
pjit_p.def_impl(_pjit_call_impl)

@cache()
def _pjit_lower(
    jaxpr: core.ClosedJaxpr,
    in_axis_resources: Tuple[CanonicalizedParsedPartitionSpec, ...],
    out_axis_resources: Tuple[CanonicalizedParsedPartitionSpec, ...],
    resource_env,
    donated_invars,
    name: str,
    in_positional_semantics, out_positional_semantics):
  # in_axis_resources and out_axis_resources are canonicalized to avoid
  # recompilation (since pjit_lower is cached) if its compiled with `None` but
  # in the next call `P(None)` is passed. Those are the same thing so should be
  # treat as equivalent and pjit_lower's cache shouldn't be invalidated.
  in_axes = [get_array_mapping(axes) for axes in in_axis_resources]
  out_axes = [get_array_mapping(axes) for axes in out_axis_resources]
  pxla.resource_typecheck(jaxpr, resource_env, {}, lambda: "pjit")
  f = core.jaxpr_as_fun(jaxpr)
  f.__name__ = name
  fun = lu.wrap_init(f)
  in_is_gda = [ips == maps._PositionalSemantics.GLOBAL
               for ips in in_positional_semantics]
  return pxla.lower_mesh_computation(
      fun, 'pjit', name, resource_env.physical_mesh,
      in_axes, out_axes, donated_invars,
      True, jaxpr.in_avals, tiling_method=None, in_is_gda=in_is_gda)


def _pjit_abstract_eval(*args, jaxpr, out_axis_resources, resource_env,
                        out_positional_semantics, **_):
  return global_to_local(out_positional_semantics, resource_env.physical_mesh,
                         jaxpr.out_avals, out_axis_resources)
pjit_p.def_abstract_eval(_pjit_abstract_eval)


def _pjit_translation_rule(ctx, avals_in, avals_out, *in_nodes, name,
                           jaxpr, in_axis_resources, out_axis_resources,
                           resource_env, donated_invars, in_positional_semantics,
                           out_positional_semantics):
  # TODO: Make this into an MLIR rule and use manual_axes!
  mesh = resource_env.physical_mesh
  subc = xc.XlaBuilder(f"pjit_{name}")

  args = []
  for i, (n, aval, axis_resources) in enumerate(
      safe_zip(in_nodes, avals_in, in_axis_resources)):
    # N.B. inlined calls shouldn't have shardings set directly on the inputs or
    # outputs (set_sharding_proto adds an identity operation).
    arg = xla.parameter(subc, i, ctx.builder.GetShape(n))
    args.append(
        xla.set_sharding_proto(
            subc, arg, get_aval_sharding_proto(aval, axis_resources, mesh)))

  # TODO: Think about how to avoid duplicating constants with the outer jaxpr
  sub_ctx = ctx.replace(
      builder=subc,
      name_stack=extend_name_stack(ctx.name_stack, wrap_name(name, "pjit")))
  out_nodes = xla.jaxpr_subcomp(
      sub_ctx, jaxpr.jaxpr, xla._xla_consts(subc, jaxpr.consts), *args)
  out_nodes = [
      xla.set_sharding_proto(subc, out,
                             get_aval_sharding_proto(aval, axis_resources, mesh))
      for out, aval, axis_resources in safe_zip(
          out_nodes, avals_out, out_axis_resources)
  ]

  subc = subc.build(xops.Tuple(subc, out_nodes))
  return xla.xla_destructure(ctx.builder,
                             xops.Call(ctx.builder, subc, list(in_nodes)))
xla.register_translation(pjit_p, _pjit_translation_rule)


def _pjit_batcher(insert_axis,
                  axis_size, axis_name, main_type,
                  vals_in, dims_in,
                  jaxpr, in_axis_resources, out_axis_resources,
                  resource_env, donated_invars, name, in_positional_semantics,
                  out_positional_semantics):
  # batch_jaxpr expects all batching dimensions to be equal to 0
  vals_in = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
             else x for x, d in zip(vals_in, dims_in)]
  is_mapped_in = [d is not batching.not_mapped for d in dims_in]
  new_jaxpr, is_mapped_out = batching.batch_jaxpr(
      jaxpr, axis_size, is_mapped_in,
      instantiate=False, axis_name=axis_name, main_type=main_type)

  new_parts = (axis_name,) if insert_axis else ()
  in_axis_resources = tuple(
      spec.insert_axis_partitions(0, new_parts) if is_mapped else spec
      for is_mapped, spec in zip(is_mapped_in, in_axis_resources))
  out_axis_resources = tuple(
      spec.insert_axis_partitions(0, new_parts) if is_mapped else spec
      for is_mapped, spec in zip(is_mapped_out, out_axis_resources))
  vals_out = pjit_p.bind(
    *vals_in,
    jaxpr=new_jaxpr,
    in_axis_resources=in_axis_resources,
    out_axis_resources=out_axis_resources,
    resource_env=resource_env,
    donated_invars=donated_invars,
    name=name,
    in_positional_semantics=in_positional_semantics,
    out_positional_semantics=out_positional_semantics)
  dims_out = [0 if batched else batching.not_mapped for batched in is_mapped_out]
  return vals_out, dims_out
batching.axis_primitive_batchers[pjit_p] = partial(_pjit_batcher, False)
pxla.spmd_primitive_batchers[pjit_p] = partial(_pjit_batcher, True)


def _pjit_jvp(primals_in, tangents_in,
              jaxpr, in_axis_resources, out_axis_resources,
              resource_env, donated_invars, name, in_positional_semantics,
              out_positional_semantics):
  is_nz_tangents_in = [type(t) is not ad.Zero for t in tangents_in]
  jaxpr_jvp, is_nz_tangents_out = ad.jvp_jaxpr(
      jaxpr, is_nz_tangents_in, instantiate=False)

  def _filter_zeros(is_nz_l, l):
    return (x for nz, x in zip(is_nz_l, l) if nz)
  _filter_zeros_in = partial(_filter_zeros, is_nz_tangents_in)
  _filter_zeros_out = partial(_filter_zeros, is_nz_tangents_out)
  outputs = pjit_p.bind(
      *primals_in, *_filter_zeros_in(tangents_in),
      jaxpr=jaxpr_jvp,
      in_axis_resources=(*in_axis_resources, *_filter_zeros_in(in_axis_resources)),
      out_axis_resources=(*out_axis_resources, *_filter_zeros_out(out_axis_resources)),
      resource_env=resource_env,
      donated_invars=(*donated_invars, *_filter_zeros_in(donated_invars)),
      name=wrap_name(name, 'jvp'),
      in_positional_semantics=(*in_positional_semantics, *_filter_zeros_in(in_positional_semantics)),
      out_positional_semantics=out_positional_semantics)

  primals_out, tangents_out = split_list(outputs, [len(jaxpr.jaxpr.outvars)])
  assert len(primals_out) == len(jaxpr.jaxpr.outvars)
  tangents_out_it = iter(tangents_out)
  return primals_out, [next(tangents_out_it) if nz else ad.Zero(aval)
                       for nz, aval in zip(is_nz_tangents_out, jaxpr.out_avals)]
ad.primitive_jvps[pjit_p] = _pjit_jvp


def _pjit_partial_eval(trace, *in_tracers,
                       jaxpr, in_axis_resources, out_axis_resources,
                       resource_env, donated_invars, name, in_positional_semantics,
                       out_positional_semantics):
  # XXX: At the moment all residuals get fully replicated, which is extremely
  #      wasteful and might quickly lead to OOM errors.
  mesh = resource_env.physical_mesh
  in_pvals = [t.pval for t in in_tracers]

  known_ins = tuple(pv.is_known() for pv in in_pvals)
  unknown_ins = tuple(not k for k in known_ins)
  raw_known_jaxpr, raw_unknown_jaxpr, unknown_outs = pe.partial_eval_jaxpr(
      jaxpr, unknown_ins, instantiate=False)
  unknown_outs = tuple(unknown_outs)
  known_outs = tuple(not uk for uk in unknown_outs)
  num_residuals = len(raw_known_jaxpr.jaxpr.outvars) - len(unknown_outs)

  def keep_where(l, should_keep):
    return tuple(x for x, keep in zip(l, should_keep) if keep)

  # Prepare the known jaxpr
  # TODO(apaszke): map_jaxpr will break caching!
  known_jaxpr = raw_known_jaxpr.map_jaxpr(lambda jaxpr: pe._drop_vars(
      jaxpr,
      drop_ins=unknown_ins,
      drop_outs=unknown_outs + (False,) * num_residuals))
  # Compute the known outputs
  known_params = dict(
      jaxpr=known_jaxpr,
      in_axis_resources=keep_where(in_axis_resources, known_ins),
      out_axis_resources=(keep_where(out_axis_resources, known_outs) +
                          (REPLICATED,) * num_residuals),
      resource_env=resource_env,
      donated_invars=keep_where(donated_invars, known_ins),
      name=name,
      in_positional_semantics=keep_where(in_positional_semantics, known_ins),
      out_positional_semantics=out_positional_semantics)

  if num_residuals:
    executable = _pjit_lower(**known_params).compile(
        _allow_propagation_to_outputs=True, _allow_compile_replicated=False)
    output_op_sharding = \
        executable.xla_executable.hlo_modules()[0].spmd_output_sharding
    output_sharding_specs = parse_op_sharding(output_op_sharding, mesh)
    residual_specs = tuple(output_sharding_specs[-num_residuals:])
  else:
    residual_specs = ()
  known_params['out_axis_resources'] = (
      keep_where(out_axis_resources, known_outs) + residual_specs)

  all_known_outs = pjit_p.bind(
      *(pv.get_known() for pv in in_pvals if pv.is_known()),
      **known_params)
  if num_residuals:
    known_out_vals, residual_vals = split_list(all_known_outs, [-num_residuals])
  else:
    known_out_vals, residual_vals = all_known_outs, ()
  known_tracers_out = [trace.new_const(known_out) for known_out in known_out_vals]
  residual_tracers = [trace.new_instantiated_const(residual) for residual in residual_vals]

  # Prepare the unknown jaxpr
  # TODO(apaszke): map_jaxpr will break caching!
  unknown_jaxpr = raw_unknown_jaxpr.map_jaxpr(lambda jaxpr: pe._drop_vars(
      jaxpr,
      drop_ins=known_ins + (False,) * num_residuals,
      drop_outs=known_outs))
  # Prepare unknown tracers
  unknown_params = dict(
      jaxpr=unknown_jaxpr,
      in_axis_resources=(keep_where(in_axis_resources, unknown_ins) + residual_specs),
      out_axis_resources=keep_where(out_axis_resources, unknown_outs),
      resource_env=resource_env,
      donated_invars=(keep_where(donated_invars, unknown_ins) +
                      (False,) * num_residuals),
      name=name,
      in_positional_semantics=(keep_where(
          in_positional_semantics, unknown_ins) + (out_positional_semantics,) * num_residuals),
      out_positional_semantics=out_positional_semantics)
  unknown_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  unknown_tracers_out = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
      for aval in global_to_local(unknown_params["out_positional_semantics"],
                                  mesh, unknown_jaxpr.out_avals,
                                  unknown_params["out_axis_resources"])
  ]
  eqn = pe.new_eqn_recipe((*unknown_tracers_in, *residual_tracers),
                          unknown_tracers_out,
                          pjit_p,
                          unknown_params,
                          source_info_util.current())
  for t in unknown_tracers_out: t.recipe = eqn
  return pe._zip_knowns(known_tracers_out, unknown_tracers_out, unknown_outs)
pe.custom_partial_eval_rules[pjit_p] = _pjit_partial_eval


def _pjit_transpose(reduce_axes, cts_in, *primals_in,
                    jaxpr, in_axis_resources, out_axis_resources,
                    resource_env, donated_invars, name, in_positional_semantics,
                    out_positional_semantics):
  mesh = resource_env.physical_mesh

  def prune_type(ty, xs, maybe_zeros):
    return tuple(x for x, mz in zip(xs, maybe_zeros) if not type(mz) is ty)

  body = lu.wrap_init(ad.closed_backward_pass)
  body = lu.hashable_partial(body, jaxpr, reduce_axes, False)
  primals_and_nz_cts_in, in_treedef = tree_flatten((primals_in, cts_in))
  body, cts_out_treedef_thunk = flatten_fun_nokwargs(body, in_treedef)

  transpose_in_axis_resources = (
    *prune_type(ad.UndefinedPrimal, in_axis_resources, primals_in),
    *prune_type(ad.Zero, out_axis_resources, cts_in)
  )
  transpose_in_positional_semantics = (
    *prune_type(ad.UndefinedPrimal, in_positional_semantics, primals_in),
    *prune_type(ad.Zero, (out_positional_semantics,) * len(cts_in), cts_in)
  )
  global_cts_in_avals = local_to_global(
      transpose_in_positional_semantics,
      mesh,
      [core.raise_to_shaped(core.get_aval(ct)) for ct in primals_and_nz_cts_in],
      transpose_in_axis_resources)
  transpose_jaxpr, global_cts_out_avals, consts = pe.trace_to_jaxpr_dynamic(
      body, global_cts_in_avals)
  # TODO(apaszke): Creating ClosedJaxpr by hand will break compilation cache!
  transpose_jaxpr = core.ClosedJaxpr(transpose_jaxpr, consts)
  del consts
  cts_out_treedef = cts_out_treedef_thunk()
  transpose_out_axis_resources = prune_type(
      ad.Zero,
      in_axis_resources,
      tree_unflatten(cts_out_treedef, [object()] * cts_out_treedef.num_leaves))

  nz_cts_out = pjit_p.bind(
      *primals_and_nz_cts_in,
      jaxpr=transpose_jaxpr,
      in_axis_resources=transpose_in_axis_resources,
      out_axis_resources=transpose_out_axis_resources,
      resource_env=resource_env,
      donated_invars=(False,) * len(primals_and_nz_cts_in),
      name=name,
      in_positional_semantics=transpose_in_positional_semantics,
      out_positional_semantics=out_positional_semantics)
  return tree_unflatten(cts_out_treedef, nz_cts_out)
ad.reducing_transposes[pjit_p] = _pjit_transpose


def _check_resources_against_named_axes(what, aval, pos_axis_resources, named_axis_resources):
  pjit_resources = set(
      it.chain.from_iterable([d for d in pos_axis_resources if d is not None]))
  aval_resources = set(it.chain.from_iterable(
    named_axis_resources[a] for a in aval.named_shape))
  overlap = pjit_resources & aval_resources
  if overlap:
    raise JAXTypeError(
        f"{what} has an axis resources specification of "
        f"{pos_axis_resources.unsynced_user_spec(SpecSync.DIM_PERMUTE)} "
        f"that uses one or more mesh axes already used by xmap to partition "
        f"a named axis appearing in its named_shape (both use mesh axes "
        f"{pxla.show_axes(overlap)})")

def _resource_typing_pjit(avals, params, source_info, resource_env, named_axis_resources):
  jaxpr = params["jaxpr"]
  what = "pjit input"
  if resource_env.physical_mesh != params['resource_env'].physical_mesh:
    raise RuntimeError("Changing the physical mesh is not allowed inside pjit.")
  for aval, pos_axis_resources in zip(jaxpr.in_avals, params['in_axis_resources']):
    _check_resources_against_named_axes(what, aval, pos_axis_resources, named_axis_resources)
  pxla.resource_typecheck(
      jaxpr.jaxpr, resource_env, named_axis_resources,
      lambda: (f"a pjit'ed function {params['name']} "
               f"(pjit called at {source_info_util.summarize(source_info)})"))
  what = "pjit output"
  for aval, pos_axis_resources in zip(jaxpr.out_avals, params['out_axis_resources']):
    _check_resources_against_named_axes(what, aval, pos_axis_resources, named_axis_resources)
pxla.custom_resource_typing_rules[pjit_p] = _resource_typing_pjit


# -------------------- with_sharding_constraint --------------------

def with_sharding_constraint(x, axis_resources):
  x_flat, tree = tree_flatten(x)
  parsed_axis_resources, entries, _ = _prepare_axis_resources(
      axis_resources, "axis_resources", allow_unconstrained_dims=True)
  axis_resources_flat = tuple(
      flatten_axes("with_sharding_constraint axis_resources",
                   tree, parsed_axis_resources))
  resource_env = pxla.thread_resources.env
  mesh = resource_env.physical_mesh
  _check_shapes_against_resources(
      "with_sharding_constraint arguments",
      mesh.is_multi_process, mesh.shape,
      x_flat, axis_resources_flat, allow_uneven_sharding=True)
  outs = [sharding_constraint_p.bind(y, axis_resources=r, resource_env=resource_env)
          for y, r in safe_zip(x_flat, axis_resources_flat)]
  return tree_unflatten(tree, outs)

def _sharding_constraint_impl(x, axis_resources, resource_env):
  # TODO(skye): can we also prevent this from being called in other
  # non-pjit contexts? (e.g. pmap, control flow)
  raise NotImplementedError(
      "with_sharding_constraint() should only be called inside pjit()")

sharding_constraint_p = core.Primitive("sharding_constraint")
sharding_constraint_p.def_impl(_sharding_constraint_impl)
sharding_constraint_p.def_abstract_eval(lambda x, **_: x)
ad.deflinear2(sharding_constraint_p,
              lambda ct, _, axis_resources, resource_env: (
                  sharding_constraint_p.bind(
                      ct, axis_resources=axis_resources, resource_env=resource_env),))

def _sharding_constraint_translation_rule(ctx, avals_in, avals_out, x_node, *,
                                          axis_resources, resource_env):
  aval, = avals_in
  mesh = resource_env.physical_mesh
  return [
      xla.set_sharding_proto(
          ctx.builder,
          x_node,
          get_aval_sharding_proto(
              aval, axis_resources, mesh, allow_uneven_axes=True),
          unspecified_dims=get_unconstrained_dims(axis_resources))
  ]
xla.register_translation(sharding_constraint_p, _sharding_constraint_translation_rule)

def _sharding_constraint_mhlo_lowering(ctx, x_node, *, axis_resources,
                                       resource_env):
  aval, = ctx.avals_in
  mesh = resource_env.physical_mesh
  return [
      mlir.wrap_with_sharding_op(
          x_node,
          get_aval_sharding_proto(
              aval,
              axis_resources,
              mesh,
              ctx.module_context.axis_context,
              allow_uneven_axes=True),
          unspecified_dims=get_unconstrained_dims(axis_resources))
  ]
mlir.register_lowering(sharding_constraint_p,
                       _sharding_constraint_mhlo_lowering)


def _sharding_constraint_batcher(insert_axis, axis_size, axis_name, main_type, vals_in, dims_in,
                                 axis_resources, resource_env):
  x, = vals_in
  d, = dims_in
  # None means unconstrained in ParsedPartitionSpec
  new_parts = (axis_name,) if insert_axis else None
  y = sharding_constraint_p.bind(
      x,
      axis_resources=axis_resources.insert_axis_partitions(d, new_parts),
      resource_env=resource_env)
  return y, d
batching.axis_primitive_batchers[sharding_constraint_p] = partial(_sharding_constraint_batcher, False)
pxla.spmd_primitive_batchers[sharding_constraint_p] = partial(_sharding_constraint_batcher, True)

def _resource_typing_sharding_constraint(avals, params, source_info, resource_env, named_axis_resources):
  aval, = avals
  _check_resources_against_named_axes(
    "with_sharding_constraint input", aval,
    params['axis_resources'], named_axis_resources)
pxla.custom_resource_typing_rules[sharding_constraint_p] = \
    _resource_typing_sharding_constraint

# -------------------- helpers --------------------

def get_array_mapping(axis_resources: ParsedPartitionSpec) -> pxla.ArrayMapping:
  return OrderedDict((axis, i)
                     for i, axes in enumerate(axis_resources)
                     if axes is not None for axis in axes)


def get_aval_sharding_proto(aval: core.AbstractValue,
                            axis_resources: ParsedPartitionSpec,
                            mesh: maps.Mesh,
                            axis_ctx: Optional[mlir.SPMDAxisContext] = None,
                            allow_uneven_axes: bool = False) -> xc.OpSharding:
  array_mapping = get_array_mapping(axis_resources)
  sharding_spec = pxla.mesh_sharding_specs(
      mesh.shape, mesh.axis_names, allow_uneven_axes=True)(aval, array_mapping)
  special_axes = {}
  if axis_ctx is not None:
    axis_names = mesh.axis_names
    for manual_axis in axis_ctx.manual_axes:
      special_axes[axis_names.index(manual_axis)] = xc.OpSharding.Type.MANUAL
  return sharding_spec.sharding_proto(special_axes=special_axes)


def get_unconstrained_dims(axis_resources: ParsedPartitionSpec):
  return {i for i, axes in enumerate(axis_resources) if axes is None}


def global_to_local(positional_semantics, mesh, avals, axes):
  if isinstance(positional_semantics, maps._PositionalSemantics):
    positional_semantics = [positional_semantics] * len(axes)
  return [
      aval if ps == maps._PositionalSemantics.GLOBAL or aval_axes.partitions == () else mesh._global_to_local(
          get_array_mapping(aval_axes), aval)
      for aval, aval_axes, ps in safe_zip(avals, axes, positional_semantics)
  ]

def local_to_global(positional_semantics, mesh, avals, axes):
  return [
      aval if ps == maps._PositionalSemantics.GLOBAL or aval_axes.partitions == () else mesh._local_to_global(
          get_array_mapping(aval_axes), aval)
      for aval, aval_axes, ps in safe_zip(avals, axes, positional_semantics)
  ]

def _create_cpspec(x):
  return x if _is_from_gda(x) else CanonicalizedParsedPartitionSpec(x)

def _maybe_replace_from_gda_with_pspec(
    in_axis_resources_flat: CanonicalizedParsedPartitionSpec, arg) -> CanonicalizedParsedPartitionSpec:
  if isinstance(arg, GDA):
    gda_cpspec = CanonicalizedParsedPartitionSpec(
        ParsedPartitionSpec.from_user_input(
            arg.mesh_axes, arg_name="GDA mesh_axes"))
    assert type(gda_cpspec) is CanonicalizedParsedPartitionSpec
    if (not _is_from_gda(in_axis_resources_flat) and
        in_axis_resources_flat != gda_cpspec):
      raise ValueError(
          'Got an input GDA to pjit with different partitioning than specified in '
          "the in_axis_resources argument to pjit. The partitioning must match, or "
          "use `jax.experimental.pjit.FROM_GDA` in `in_axis_resources`. "
          f"Got GDA spec: {gda_cpspec.user_spec} and "
          f"pjit spec: {in_axis_resources_flat.user_spec} for GDA: {arg}")
    return gda_cpspec
  return in_axis_resources_flat


def _maybe_check_pjit_gda_mesh(args, mesh):
  for x in args:
    if isinstance(x, GDA) and x.mesh != mesh:
      raise ValueError("Pjit's mesh and GDA's mesh should be equal. Got Pjit "
                       f"mesh: {mesh},\n GDA mesh: {x.mesh}")

# -------------------- XLA OpSharding to PartitionSpec --------------------
# Note that OpSharding is more expressive than PartitionSpecs, so it's not
# always possible to convert them, but the code below should at least
# support handle all cases when this is possible.

def strides_for_sizes(sizes):
  """Returns an array of strides for major-to-minor sizes."""
  return np.cumprod(sizes[::-1])[::-1] // np.asarray(sizes)

def unflatten_array(named_sizes, assignment):
  """Recovers the ordering of axis names based on a device assignment.

  The device assignments that this function can convert into axis orders
  are of the form::

    np.arange(np.prod(named_sizes.values())).transpose(...).flatten()

  for some transposition ``...``. This is satisfied by all OpSharding assignments
  generated from partition specs.

  Arguments:
    named_sizes: A dictionary mapping axis names to their sizes.
    assignment: A permutation of integers between 0 and the product of all
      named sizes.

  Returns:
    A major-to-minor list of axis names that corresponds to the given assignment.
  """
  named_sizes = {name: size for name, size in named_sizes.items() if size != 1}
  sizes = np.fromiter(named_sizes.values(), dtype=np.int64)
  strides = strides_for_sizes(sizes)
  dims = explode_superdims(sizes, unflatten_superdims(assignment))
  dim_to_name = {(size, stride): name for size, stride, name in zip(sizes, strides, named_sizes)}
  return [dim_to_name[d] for d in dims]

def unflatten_superdims(assignment):
  """Unflatten a list of dimension sizes and their strides that generates assignment.

  If this function succeeds for a given ``assignment``, then the following property
  should be satisfied::

    dims_with_strides = unflatten_superdims(assignment)
    base_array = np.arange(map(fst, sorted(dims_with_strides, key=snd, reverse=True)))
    assignment == base_array.transpose(argsort(dims_with_strides, key=snd, reverse=True)).flatten()

  That is, the returned dimensions list all sizes of the base array (with strides
  indicating their initial order). The order of dimensions in the list corresponds
  to the permutation that applied to the base array generates the assignment.
  """
  def check(cond):
    if cond: return
    raise NotImplementedError("Failed to convert OpSharding into a ShardingSpec. "
                              "Please open a bug report!")
  flat_assignment = np.asarray(assignment, dtype=np.int64)
  check(flat_assignment[0] == 0)
  dims = []
  while flat_assignment.size > 1:
    stride = flat_assignment[1]
    for i in range(len(flat_assignment)):
      if flat_assignment[i] != i * stride: break
    else:
      # After this loop i should point to an "element after the sequence", so
      # we have to increment it if the whole array is a strided sequence.
      i += 1
    size = i
    dims.append((size, stride))
    assert size > 1  # Ensure progress
    flat_assignment = flat_assignment[::size]
  return dims

def explode_superdims(sizes, dims):
  """Explode superdims to fit a known shape.

  The unflattening process might mistakenly generate too few too large dimensions.
  For example, ``unflatten_superdims(np.arange(n))`` always returns ``[(n, 1)]``.
  This function takes a list of such contiguous super-dimensions and splits them
  into smaller dimensions such that::

    set(map(fst, explode_superdims(sizes, dims))) == set(sizes)
  """
  strides_to_sizes = {stride: size for size, stride in zip(sizes, strides_for_sizes(sizes))}
  dims = list(reversed(dims))
  final_dims = []
  for size, stride in dims:
    target_size = strides_to_sizes[stride]
    new_dims = []
    while size > target_size:
      assert target_size > 1  # Ensure progress
      assert size % target_size == 0
      new_dims.append((target_size, stride))
      size //= target_size
      stride *= target_size
      target_size = strides_to_sizes[stride]
    assert size == target_size
    new_dims.append((size, stride))
    final_dims += reversed(new_dims)
  return final_dims

def parse_op_sharding(op_sharding, mesh):
  if op_sharding.type == xc.OpSharding.Type.TUPLE:
    return [parse_op_sharding(s, mesh) for s in op_sharding.tuple_shardings]
  elif op_sharding.type == xc.OpSharding.Type.REPLICATED:
    return REPLICATED
  elif op_sharding.type == xc.OpSharding.Type.OTHER:
    mesh_shape = mesh.shape
    mesh_axis_order = unflatten_array(mesh.shape, op_sharding.tile_assignment_devices)
    mesh_axis = iter(mesh_axis_order)
    shape = op_sharding.tile_assignment_dimensions
    partitions = []
    for dim_size in shape:
      dim_partitions = []
      while dim_size > 1:
        axis = next(mesh_axis)
        axis_size = mesh_shape[axis]
        assert dim_size % axis_size == 0
        dim_size //= axis_size
        dim_partitions.append(axis)
      partitions.append(tuple(dim_partitions))
    if op_sharding.last_tile_dims == [xc.OpSharding.Type.REPLICATED]:
      replicate_on_last_tile_dim = True
    else:
      replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim
      if op_sharding.last_tile_dims:
        raise NotImplementedError("Unhandled OpSharding type. Please open a bug report!")
    if replicate_on_last_tile_dim:
      partitions = partitions[:-1]
    return ParsedPartitionSpec('<internally generated spec>', partitions)
  else:
    raise AssertionError("Unhandled OpSharding type. Please open a bug report!")
