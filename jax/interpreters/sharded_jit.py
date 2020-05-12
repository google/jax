# Copyright 2020 Google LLC
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

from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .. import core
from . import ad
from . import partial_eval as pe
# TODO(skye): separate pmap into it's own module?
from . import pxla
from . import xla
from .. import linear_util as lu
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..api_util import flatten_axes, flatten_fun
from ..tree_util import tree_flatten, tree_unflatten
from ..util import extend_name_stack, wrap_name, safe_zip

xops = xc._xla.ops


def _map(f, *xs):
  return tuple(map(f, *xs))


class ResultToPopulate: pass
result_to_populate = ResultToPopulate()

def _pvals_to_results_handler(nrep, npart, partitions, out_pvals):
  nouts = len(out_pvals)
  handlers = [_pval_to_result_handler(npart, parts, out_pval)
              for parts, out_pval in safe_zip(partitions, out_pvals)]

  def handler(out_bufs):
    assert nrep * npart == len(out_bufs)
    buffers = [[result_to_populate] * nrep * npart for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf):
        buffers[i][r] = buf
    assert not any(buf is result_to_populate for bufs in buffers
                   for buf in bufs)
    return [h(bufs) for h, bufs in zip(handlers, buffers)]

  return handler


def _pval_to_result_handler(npart, parts, pval):
  pv, const = pval
  if pv is None:
    raise NotImplementedError  # TODO(skye): handle constant outputs
  else:
    if pv is not core.abstract_unit:
      spec = _partitioned_sharding_spec(npart, parts, pv)
      indices = pxla.spec_to_indices(pv.shape, spec)
    else:
      spec = indices = None
    return pxla.aval_to_result_handler(spec, indices, pv)


@lu.cache
def _sharded_callable(
    fun: lu.WrappedFun, num_partitions: Optional[int],
    in_parts: Tuple[Optional[Tuple[int, ...]], ...],
    out_parts_thunk: Callable[[], Tuple[Optional[Tuple[int, ...]], ...]],
    name: str, *abstract_args):
  nrep = 1
  in_pvals = [pe.PartialVal.unknown(aval) for aval in abstract_args]
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(fun, in_pvals, instantiate=False, bottom=True)

  # TODO(skye): add tests for equationless jaxpr cases
  if not jaxpr.eqns and all(outvar.aval is core.abstract_unit
                            for outvar in jaxpr.outvars):
    return lambda *_: [
        const if pv is None else core.unit for pv, const in out_pvals
    ]

  if xb.get_backend().platform != "tpu":
    # TODO(skye): fall back to regular jit?
    raise ValueError("sharded_jit only works on TPU!")

  num_inner_parts = _inner_partitions(jaxpr, num_partitions)
  if num_partitions is None:
    num_partitions = num_inner_parts
  assert num_partitions == num_inner_parts
  if num_partitions is None:
    # No partitions specified anywhere, everything is replicated.
    num_partitions = 1

  out_parts = out_parts_thunk()

  c = xb.make_computation_builder("spjit_{}".format(fun.__name__))
  xla_consts = _map(partial(xb.constant, c), consts)
  xla_args = _xla_sharded_args(c, abstract_args, in_parts)
  axis_env = xla.AxisEnv(nrep, (), ())
  out_nodes = xla.jaxpr_subcomp(
      c, jaxpr, None, axis_env, xla_consts,
      extend_name_stack(wrap_name(name, "sharded_jit")), *xla_args)
  out_tuple = xb.with_sharding(c, out_parts, xops.Tuple, c, out_nodes)
  built = c.Build(out_tuple)

  devices = xb.local_devices()[:num_partitions]
  assert len(devices) == num_partitions
  device_assignment = np.array([[d.id for d in devices]])
  device_assignment = np.reshape(device_assignment, (-1, num_partitions))
  # device_assignment = None  # TODO(skye): replace with default device assignment?

  compiled = xb.get_backend().compile(
      built, compile_options=xb.get_compile_options(
          nrep, num_partitions, device_assignment))

  input_specs = [
      _partitioned_sharding_spec(num_partitions, parts, aval)
      for parts, aval in zip(in_parts, abstract_args)]
  input_indices = [pxla.spec_to_indices(aval.shape, spec)
                   if spec is not None else None
                   for aval, spec in zip(abstract_args, input_specs)]

  handle_args = partial(pxla.shard_args, compiled.local_devices(),
                        input_indices)
  handle_outs = _pvals_to_results_handler(nrep, num_partitions, out_parts,
                                          out_pvals)
  return partial(_execute_spatially_partitioned, compiled, handle_args,
                 handle_outs)


def _partitioned_sharding_spec(num_partitions: int,
                               partitions: Optional[Sequence[int]], aval):
  if aval is core.abstract_unit:
    return None

  if partitions is None:
    return pxla.ShardingSpec(
        # int(1) because pytype is confused by 1 (???)
        shards_per_axis=(int(1),) * len(aval.shape),
        is_axis_materialized=(True,) * len(aval.shape),
        replication_factor=num_partitions)
  else:
    assert len(partitions) == len(aval.shape)
    return pxla.ShardingSpec(
        shards_per_axis=tuple(partitions),
        is_axis_materialized=(True,) * len(aval.shape),
        replication_factor=1)


def _sharded_jit_translation_rule(c, axis_env, in_nodes, name_stack,
                                  in_parts, out_parts_thunk, num_partitions,
                                  backend, name, call_jaxpr):
  subc = xc.XlaBuilder(f"sharded_jit_{name}")

  # We assume any extra leading in_nodes are constants and replicate them.
  num_extra_nodes = len(in_nodes) - len(in_parts)
  assert num_extra_nodes >= 0
  in_parts = (None,) * num_extra_nodes + in_parts

  args = []
  for i, (n, sharding) in enumerate(safe_zip(in_nodes, in_parts)):
    # We use xb.set_sharding instead of xb.with_sharding because inlined calls
    # shouldn't have shardings set directly on the inputs or outputs.
    arg = xb.parameter(subc, i, c.GetShape(n))
    args.append(xb.set_sharding(subc, arg, sharding))

  out_nodes = xla.jaxpr_subcomp(
      subc, call_jaxpr, backend, axis_env, (),
      extend_name_stack(name_stack, wrap_name(name, "sharded_jit")), *args)
  out_parts = out_parts_thunk()
  assert len(out_parts) == len(out_nodes)
  out_nodes = [xb.set_sharding(subc, out, sharding)
               for out, sharding in safe_zip(out_nodes, out_parts)]

  subc = subc.build(xops.Tuple(subc, out_nodes))
  return xops.Call(c, subc, list(in_nodes))


def _execute_spatially_partitioned(compiled, in_handler, out_handler, *args):
  input_bufs = in_handler(args)
  out_bufs = compiled.execute_on_local_devices(list(input_bufs))
  return out_handler(out_bufs)


def _xla_sharded_args(c, avals, in_parts):
  xla_args = []
  for i, (sharding, aval) in enumerate(safe_zip(in_parts, avals)):
    param = xb.with_sharding(c, sharding, xb.parameter, c, i,
                             xla.aval_to_xla_shape(aval))
    xla_args.append(param)
  return xla_args


def get_num_partitions(*partitions):
  partition_specs = tree_flatten(partitions)[0]
  if len(partition_specs) == 0:
    # Everything is specified as replicated (all Nones).
    return None
  num_partitions_set = set(np.prod(spec) for spec in partition_specs)
  if len(num_partitions_set) > 1:
    raise ValueError(
        f"All partition specs must use the same number of total partitions, "
        f"got {partitions}, with distinct number of partitions "
        f"{num_partitions_set} (the total number of partitions is the product "
        f"of a partition spec)")
  assert len(num_partitions_set) == 1
  return num_partitions_set.pop()


def _inner_partitions(jaxpr, expected_num_parts: Optional[int]):
  """Returns the total number of partitions from PartitonSpecs inside `jaxpr`.

  Also validates that this number matches `expected_num_parts` if provided.
  """
  for eqn in jaxpr.eqns:
    if eqn.primitive == set_sharding_p:
      parts = eqn.params["partitions"]
      nparts = get_num_partitions(parts)
      if expected_num_parts is None:
        expected_num_parts = nparts
      elif nparts != expected_num_parts:
        # TODO(skye): raise this error as we trace the jaxpr
        raise ValueError(
            f"set_sharding with partitions={parts} (total partitions: {nparts})"
            f" doesn't match expected number of partitions: "
            f"{expected_num_parts}. If these partitions look right, check "
            f"outer sharded_jit and/or other set_sharding calls.")
      elif eqn.primitive.call_primitive:
        expected_num_parts = _inner_partitions(eqn.params["call_jaxpr"],
                                               expected_num_parts)
      # TODO(skye): control flow
  return expected_num_parts

def _sharded_call_impl(fun, *args, num_partitions, in_parts, out_parts_thunk,
                       name):
  compiled_fun = _sharded_callable(fun, num_partitions, in_parts,
                                   out_parts_thunk, name,
                                   *map(xla.abstractify, args))
  return compiled_fun(*args)


sharded_call_p = core.Primitive("sharded_call")
sharded_call_p.call_primitive = True
sharded_call_p.multiple_results = True
sharded_call = partial(core.call_bind, sharded_call_p)
sharded_call_p.def_custom_bind(sharded_call)
sharded_call_p.def_impl(_sharded_call_impl)
xla.call_translations[sharded_call_p] = _sharded_jit_translation_rule


class PartitionSpec(tuple):
  """Tuple of integer specifying how a value should be partitioned.

  Each integer corresponds to how many ways a dimension is partitioned. We
  create a separate class for this so JAX's pytree utilities can distinguish it
  from a tuple that should be treated as a pytree.
  """
  def __new__(cls, *partitions):
    return tuple.__new__(PartitionSpec, partitions)

  def __repr__(self):
    return "PartitionSpec%s" % tuple.__repr__(self)


def sharded_jit(fun: Callable, in_parts, out_parts):
  """Like ``jit``, but partitions ``fun`` across multiple devices.

  WARNING: this feature is still under active development! It may not work well,
  and may change without warning!

  `sharded_jit` sets up ``fun`` for just-in-time compilation with XLA, but
  unlike ``jit``, the compiled function will run across multiple devices
  (e.g. multiple GPUs or multiple TPU cores). This is achieved by spatially
  partitioning the data that flows through the computation, so each operation is
  run across all devices and each device runs only a shard of the full
  data. (Some data can optionally be replicated, which is sometimes more
  efficient for small arrays when combined with larger spatially-partitioned
  arrays.) Communication between devices is automatically inserted as necessary.

  ``sharded_jit`` can be useful if the jitted version of ``fun`` would not fit
  in a single device's memory, or to speed up ``fun`` by running each operation
  in parallel across multiple devices.

  Note: ``sharded_jit`` is currently available on TPU only!

  Args:
    fun: Function to be jitted.
    in_parts: The input partitions, i.e. how each argument to ``fun`` should be
      partitioned or replicated. This should be a PartitionSpec indicating into
      how many partitions each dimension should be sharded, None indicating
      replication, or (nested) standard Python containers thereof. For example,
      ``in_parts=PartitionSpec(2,1)`` means all arguments should be partitioned
      over two devices across the first dimension;
      ``in_parts=(PartitionSpec(2,2), PartitionSpec(4,1), None)`` means the
      first argument should be partitioned over four devices by splitting the
      first two dimensions in half, the second argument should be partitioned
      over the four devices across the first dimension, and the third argument
      is replicated across the four devices. All PartitionSpecs in a given
      ``sharded_jit`` call must correspond to the same total number of
      partitions, i.e. the product of all PartitionSpecs must be equal.
    out_parts: The output partitions, i.e. how each output of ``fun`` should be
      partitioned or replicated. This follows the same convention as
     ``in_parts``.

  Returns:
    A version of ``fun`` that will be distributed across multiple devices.
  """
  num_parts = get_num_partitions(in_parts, out_parts)

  def wrapped(*args, **kwargs):
    if kwargs:
      raise NotImplementedError("sharded_jit over kwargs not yet supported")
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten((args, kwargs))
    in_parts_flat = tuple(flatten_axes(in_tree.children()[0], in_parts))
    flat_fun, out_tree = flatten_fun(f, in_tree)
    # TODO(skye): having a function-typed param in a primitive seems dicey, is
    # there a better way?
    out_parts_thunk = lambda: tuple(flatten_axes(out_tree(), out_parts))
    out = sharded_call(
        flat_fun,
        *args_flat,
        num_partitions=num_parts,
        in_parts=in_parts_flat,
        out_parts_thunk=out_parts_thunk,
        name=flat_fun.__name__)
    return tree_unflatten(out_tree(), out)

  return wrapped


def _set_sharding_impl(x, partitions):
  # TODO(skye): can we also prevent this from being called in other
  # non-sharded_jit contexts? (e.g. pmap, control flow)
  raise NotImplementedError(
      "set_sharding() should only be called inside sharded_jit()")

def _set_sharding_translation_rule(c, x_node, partitions):
  return xb.set_sharding(c, x_node, partitions)

set_sharding_p = core.Primitive("set_sharding")
set_sharding_p.def_impl(_set_sharding_impl)
set_sharding_p.def_abstract_eval(lambda x, partitions: x)
ad.deflinear(set_sharding_p, lambda ct, partitions: (set_sharding(ct, partitions),))
xla.translations[set_sharding_p] = _set_sharding_translation_rule

def set_sharding(x, partitions: Optional[PartitionSpec]):
  """Identity-like function that specifies how ``x`` should be sharded.

  WARNING: this feature is still under active development! It may not work well,
  and may change without warning!

  This should only be called inside a function transformed by ``sharded_jit``.
  It refines how ``f`` is sharded. ``partitions`` must correspond to the same
  number of total partitions dictated by the outer ``sharded_jit`` and any other
  ``set_sharding`` calls. In the case where only replication has been specified,
  any ``partitions`` are valid.

  Example usage:
    @partial(sharded_jit, in_parts=None, out_parts=None, num_shards=2
    def f(x):
      y = x + 1
      y = set_sharding(y, PartitionSpec(2,1))
      return y * 2

  In this example, the inputs and outputs of ``f`` will be replicated, but the
  inner value of ``y`` will be partitioned in half. ``f`` will run on two
  devices due to the set_sharding call.

  Args:
    x: Array value
    partitions: PartitionSpec indicating how ``x`` should be partitioned, or
      None for replication.

  Returns:
    A new version of ``x`` with the specified sharding applied.
  """
  return set_sharding_p.bind(x, partitions=partitions)
