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
"""Microbenchmarks for JAX `api` functions."""

import enum
import functools
import math
import operator

import google_benchmark
import jax
from jax import lax
from jax._src.api_util import shaped_abstractify  # technically not an api fn
from jax._src.ad_checkpoint import checkpoint  # new jax.remat implementation
from jax._src.lib import xla_client as xc
from jax.interpreters import xla
from jax._src import array
from jax._src import op_shardings
from jax._src.pjit import pjit_check_aval_sharding
from jax.experimental import pjit as pjit_lib
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


partial = functools.partial

def required_devices(num_devices_required):
  """Helper to skip benchmarks that require more devices."""
  def helper1(f):
    @functools.wraps(f)
    def helper2(state):
      if jax.device_count() < num_devices_required:
        state.skip_with_error(f"requires {num_devices_required} devices")
        return
      return f(state)
    return helper2
  return helper1


def create_mesh(shape, axis_names, state):
  size = math.prod(shape)
  if len(jax.devices()) < size:
    state.skip_with_error(f"Requires {size} devices")
    return None
  devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
  mesh_devices = np.array(devices[:size]).reshape(shape)
  global_mesh = jax.sharding.Mesh(mesh_devices, axis_names)
  return global_mesh


def swap(a, b):
  return b, a


class AnEnum(enum.IntEnum):
  A = 123
  B = 456

@google_benchmark.register
def eager_unary_dispatch(state):
  a = jax.device_put(1)
  lax.neg(a)
  while state:
    lax.neg(a)


@google_benchmark.register
def eager_unary(state):
  a = jax.device_put(1)
  lax.neg(a).block_until_ready()
  while state:
    lax.neg(a).block_until_ready()


@google_benchmark.register
def eager_binary_dispatch(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  lax.add(a, b)
  while state:
    lax.add(a, b)


@google_benchmark.register
def eager_binary(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  lax.add(a, b).block_until_ready()
  while state:
    lax.add(a, b).block_until_ready()


@google_benchmark.register
def jit_trivial_dispatch(state):
  """Benchmarks only the duration for jitted_f to return the future."""
  f = jax.jit(swap)
  a, b = f(1, 2)
  x = f(a, b)
  while state:
    x = f(a, b)
  x[0].block_until_ready()


@google_benchmark.register
def jit_trivial(state):
  f = jax.jit(swap)
  a, b = f(1, 2)
  f(a, b)

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
def jit_simple_dispatch(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  f = jax.jit(operator.add)
  f(a, b)

  while state:
    f(a, b)


@google_benchmark.register
def jit_simple(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  f = jax.jit(operator.add)
  f(a, b)

  while state:
    f(a, b).block_until_ready()

@google_benchmark.register
def jit_simple_dispatch_array(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  f = jax.jit(operator.add)
  f(a, b)

  while state:
    f(a, b)


@google_benchmark.register
def jit_simple_array(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  f = jax.jit(operator.add)
  f(a, b)

  while state:
    f(a, b).block_until_ready()


@google_benchmark.register
def jit_small_matmul(state):
  x = np.random.uniform(size=(2, 2)).astype(np.float32)
  x = jax.device_put(x)

  f = jax.jit(lambda x: jnp.dot(x, x))
  f(x).block_until_ready()

  while state:
    f(x).block_until_ready()


@google_benchmark.register
def jit_big_matmul(state):
  x = np.random.uniform(size=(100, 100)).astype(np.float32)
  x = jax.device_put(x)

  f = jax.jit(lambda x: jnp.dot(x, x))
  f(x).block_until_ready()

  while state:
    f(x).block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([10])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
@google_benchmark.option.args([100])
@google_benchmark.option.args([1000])
@google_benchmark.option.args([1000])
@google_benchmark.option.args([2000])
@google_benchmark.option.args([2000])
def jit_simple_many_args_dispatch(state):
  args = [jax.device_put(i) for i in range(state.range(0))]
  f = jax.jit(lambda xs: functools.reduce(operator.add, xs))
  x = f(args)
  x.block_until_ready()

  while state:
    x = f(args)
  x.block_until_ready()

@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([10])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
@google_benchmark.option.args([100])
@google_benchmark.option.args([1000])
@google_benchmark.option.args([1000])
@google_benchmark.option.args([2000])
@google_benchmark.option.args([2000])
def jit_simple_many_args(state):
  args = [jax.device_put(i) for i in range(state.range(0))]
  f = jax.jit(lambda xs: functools.reduce(operator.add, xs))
  f(args).block_until_ready()

  while state:
    f(args).block_until_ready()

def jit_simple_pruned_args_dispatch(n, state):
  args = [jax.device_put(i) for i in range(n)]
  f = jax.jit(lambda *xs: xs[0] + 1)
  x = f(*args)
  x.block_until_ready()

  while state:
    x = f(*args)
  x.block_until_ready()


def jit_simple_pruned_args(n, state):
  args = [jax.device_put(i) for i in range(n)]
  f = jax.jit(lambda *xs: xs[0] + 1)
  x = f(*args)
  x.block_until_ready()

  while state:
    f(*args).block_until_ready()

benchmarks = []
for n in [10, 100, 1000, 2000]:
  benchmarks += [
      google_benchmark.register(partial(jit_simple_pruned_args_dispatch, n),
                                name=f"jit_simple_pruned_args_dispatch_{n}"),
      google_benchmark.register(partial(jit_simple_pruned_args, n),
                                name=f"jit_simple_pruned_args_{n}")
  ]


@google_benchmark.register
def jit_dispatch_without_transfer(state):
  # We pick up a realistic input. 224 is usual for classification and 128 a
  # TPU-friendly batch-size.
  imgs = np.ones((128, 224, 224), np.float32)
  imgs = jax.device_put(imgs)

  f = jax.jit(lambda x: x+1)
  f(imgs)

  while state:
    f(imgs)


@google_benchmark.register
def jit_dispatch_with_transfer(state):
  imgs = np.ones((128, 224, 224), np.float32)

  f = jax.jit(lambda x: x+1)
  f(imgs).block_until_ready()

  while state:
    x = f(imgs)
  x.block_until_ready()


@google_benchmark.register
@required_devices(2)
def pmap_trivial_2_devices(state):
  f = jax.pmap(swap)
  a, b = f(jnp.array([1, 2]), jnp.array([3, 4]))

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
@required_devices(8)
def pmap_trivial_dispatch_8_devices(state):
  f = jax.pmap(swap)
  a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
           jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

  while state:
    a, b = f(a, b)


@google_benchmark.register
@required_devices(8)
def pmap_trivial_8_devices(state):
  f = jax.pmap(swap)
  a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
           jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
@required_devices(2)
def pmap_simple_2_devices(state):
  f = jax.pmap(lambda a, b: (a + b, a - b))
  a, b = f(jnp.array([1, 2]), jnp.array([3, 4]))

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
@required_devices(8)
def pmap_simple_dispatch_8_devices(state):
  f = jax.pmap(lambda a, b: (a + b, a - b))
  a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
           jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

  while state:
    a, b = f(a, b)


@google_benchmark.register
@required_devices(8)
def pmap_simple_8_devices(state):
  f = jax.pmap(lambda a, b: (a + b, a - b))
  a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
           jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
@required_devices(8)
def pmap_simple_dispatch_8_devices_100_args(state):
  f = jax.pmap(lambda *args: args[1:] + (args[0] + 1,))
  args = []
  for i in range(100):
    args.append(jnp.array(list(range(i, i+8))))

  args = f(*args)

  while state:
    args = f(*args)


@google_benchmark.register
@required_devices(8)
def pmap_simple_8_devices_100_args(state):
  f = jax.pmap(lambda *args: args[1:] + (args[0] + 1,))
  args = []
  for i in range(100):
    args.append(jnp.array(list(range(i, i+8))))

  # Warmup loop.
  out = f(*args)

  while state:
    out = f(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)


def _run_sda_index_bench(state, num_devices):
  x = jax.pmap(jnp.sin)(jnp.arange(num_devices))
  jax.device_get(x)
  while state:
    for i in range(num_devices):
      _ = x[i]


@google_benchmark.register
@required_devices(1)
def sda_index_1(state):
  _run_sda_index_bench(state, 1)


@google_benchmark.register
@required_devices(2)
def sda_index_2(state):
  _run_sda_index_bench(state, 2)


@google_benchmark.register
@required_devices(8)
def sda_index_8(state):
  _run_sda_index_bench(state, 8)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_shaped_abstractify(state):
  device, *_ = jax.devices()
  args = [jax.device_put_replicated(1, [device])] * 1000
  while state:
    _ = [shaped_abstractify(x) for x in args]


def _run_benchmark_for_xla_abstractify(arg, state):
  while state:
    xla.abstractify(arg)

def bench_xla_abstractify():
  _abstractify_args = [
      (3, 'scalar_int'),
      (3.5, 'scalar_float'),
      (np.int32(3), 'scalar_numpy_int32'),
      (np.uint32(7), 'scalar_numpy_uint32'),
      (np.random.randn(3, 4, 5, 6), 'numpy_random'),
      (np.arange(100, dtype=np.float32), 'numpy_arange_100_float32'),
      (AnEnum.B, 'enum'),
  ]
  benchmarks = []
  for a, name in _abstractify_args:
    benchmarks.extend([
        google_benchmark.register(
            partial(_run_benchmark_for_xla_abstractify, a),
            name=f'bench_xla_abstractify_{name}'),
    ])
bench_xla_abstractify()


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMicrosecond)
def bench_are_op_shardings_equal(state):
  op1 = xc.OpSharding()
  op1.type = xc.OpSharding.Type.OTHER
  op1.tile_assignment_dimensions = [4, 192, 16]
  op1.tile_assignment_devices = list(range(12288))

  op2 = xc.OpSharding()
  op2.type = xc.OpSharding.Type.OTHER
  op2.tile_assignment_dimensions = [4, 192, 16]
  op2.tile_assignment_devices = list(range(12288))

  while state:
    op_shardings.are_op_shardings_equal(op1, op2)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_pjit_check_aval_sharding(state):
  mesh = create_mesh((4, 2), ('x', 'y'), state)
  if mesh is None:
    return
  s = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x', 'y'))
  aval = jax.core.ShapedArray((8, 2), np.int32)

  while state:
    pjit_check_aval_sharding([s] * 100, [aval] * 100, None, 'benchmark', False)


@google_benchmark.register
def bench_addressable_shards_index(state):
  mesh = create_mesh((4, 2), ('x', 'y'), state)
  if mesh is None:
    return
  shape = (8, 2)
  inp = np.arange(math.prod(shape)).reshape(shape)
  s = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x', 'y'))
  arr = jax.device_put(inp, s)

  while state:
    [s.index for s in arr.addressable_shards]


@google_benchmark.register
def bench_addressable_shards_replica_id(state):
  mesh = create_mesh((32, 16), ('x', 'y'), state)
  if mesh is None:
    return
  shape = (64, 32)
  inp = np.arange(math.prod(shape)).reshape(shape)
  s = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x', 'y'))
  arr = jax.device_put(inp, s)

  while state:
    [s.replica_id for s in arr.addressable_shards]


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_remat_eager_retracing_overheads(state):
  def double_compose(f):
    return lambda x: f(f(x))

  f = jnp.sin
  for _ in range(6):
    f = double_compose(f)
  f = double_compose(checkpoint(f))

  while state:
    y, _ = jax.vjp(f, 3.)
  y.block_until_ready()

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_remat_eager_retracing_overheads_static_argnums(state):
  def double_compose(f):
    return lambda x, y: f(f(x, y), y)

  f = lambda x, _: jnp.sin(x)
  for _ in range(6):
    f = double_compose(f)
  f = double_compose(checkpoint(f, static_argnums=(1,)))

  while state:
    y, _ = jax.vjp(f, 3., True)
  y.block_until_ready()


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_slicing_compilation(state):
  x = jnp.arange(3)
  while state:
    jax.jit(lambda x: (x[0], x[1], x[2])).lower(x).compile()

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_slicing_compilation2(state):
  x = jnp.arange(3)
  while state:
    jax.jit(lambda x: (x[:1], x[1:2], x[2:3])).lower(x).compile()

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_repeated_static_indexing(state):
  x = jnp.arange(500)
  while state:
    jax.block_until_ready([x[i] for i in range(500)])

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_repeated_static_slicing(state):
  x = jnp.arange(1000)
  while state:
    jax.block_until_ready([x[i:i + 2] for i in range(0, 1000, 2)])

def pjit_simple_benchmark(state, num_devices, num_args, cpp_jit, use_aot=False):
  spec = jax.sharding.PartitionSpec('x')
  mesh = create_mesh((num_devices,), ('x',), state)
  if mesh is None:
    return
  s = jax.sharding.NamedSharding(mesh, spec)
  inp_data = np.arange(num_devices).astype(np.float32)
  x = array.make_array_from_callback(inp_data.shape, s, lambda idx: inp_data[idx])

  x = [x for _ in range(num_args)]

  in_axis_resources = jax.sharding.NamedSharding(mesh, spec)
  out_axis_resources = jax.sharding.NamedSharding(mesh, spec)

  f = pjit_lib.pjit(
      lambda x: jax.tree.map(lambda x: x + 1, x),
      in_shardings=in_axis_resources,
      out_shardings=out_axis_resources,
  )

  if use_aot:
    f = f.lower(x).compile()

  x = f(x)

  while state:
    x = f(x)


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([1])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
def pjit_simple_1_device(state):
  pjit_simple_benchmark(
      state, num_devices=1, num_args=state.range(0), cpp_jit=state.range(1))

@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([1])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
def pjit_simple_4_device(state):
  pjit_simple_benchmark(
      state, num_devices=4, num_args=state.range(0), cpp_jit=state.range(1))

@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([1])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
def pjit_simple_4000_device(state):
  pjit_simple_benchmark(
      state, num_devices=4000, num_args=state.range(0), cpp_jit=state.range(1))


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([1])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
def pjit_aot_1_device(state):
  pjit_simple_benchmark(
      state,
      num_devices=1,
      num_args=state.range(0),
      cpp_jit=state.range(1),
      use_aot=True)


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([1])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
def pjit_aot_4_device(state):
  pjit_simple_benchmark(
      state,
      num_devices=4,
      num_args=state.range(0),
      cpp_jit=state.range(1),
      use_aot=True)


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([1])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
def pjit_aot_4000_device(state):
  pjit_simple_benchmark(
      state,
      num_devices=4000,
      num_args=state.range(0),
      cpp_jit=state.range(1),
      use_aot=True)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def host_local_array_to_global_array(state):
  global_mesh = create_mesh((4, 2), ('x', 'y'), state)
  input_shape = (8, 2)
  input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
  in_pspec = jax.sharding.PartitionSpec('x', 'y')

  while state:
    multihost_utils.host_local_array_to_global_array(
        (input_data, input_data), global_mesh, (in_pspec, in_pspec))


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([1])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
@google_benchmark.option.args([1000])
def device_put_from_numpy_array(state):
  x = [np.array(1, np.int32)] * state.range(0)
  while state:
    _ = jax.block_until_ready(jax.device_put(x))


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args'])
@google_benchmark.option.args([1])
@google_benchmark.option.args([10])
@google_benchmark.option.args([100])
@google_benchmark.option.args([1000])
def device_put_from_jax_array(state):
  x = [np.array(1, np.int32)] * state.range(0)
  x = jax.block_until_ready(jax.device_put(x, device=jax.devices()[0]))
  d = jax.devices()[1]
  while state:
    _ = jax.block_until_ready(jax.device_put(x, device=d))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def device_put_big(state):
  x = np.arange(4000 * 10**6 // np.dtype('float32').itemsize, dtype=np.float32)
  jax.device_put(x).block_until_ready()

  while state:
    _ = jax.device_put(x).block_until_ready()


@google_benchmark.register
def device_put_sharded(state):
  arr_inp = [np.array(i) for i in range(jax.device_count())]
  dev = jax.devices()

  while state:
    _ = jax.device_put_sharded(arr_inp, dev).block_until_ready()


@google_benchmark.register
@required_devices(8)
def device_get_8_devices(state):
  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:8]).reshape((4, 2)), ('x', 'y')
  )
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('x', 'y')
  )
  inp = jax.device_put(np.zeros((8, 4), dtype=np.float32), sharding)

  @jax.jit
  def fn(x):
    y = x + x
    return [y for _ in range(50)]

  jax.device_get(fn(inp))

  while state:
    jax.device_get(fn(inp))


@google_benchmark.register
@required_devices(8)
def np_asarray_8_devices(state):
  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:8]).reshape((4, 2)), ('x', 'y')
  )
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('x', 'y')
  )
  inp = jax.device_put(np.zeros((8, 4), dtype=np.float32), sharding)

  @jax.jit
  def fn(x):
    y = x + x
    return [y for _ in range(50)]

  jax.device_get(fn(inp))

  while state:
    [np.asarray(x) for x in fn(inp)]


@google_benchmark.register
@required_devices(8)
def jax_array_arrays_8_devices(state):
  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:8]).reshape((4, 2)), ('x', 'y')
  )
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('x', 'y')
  )
  inp = jax.device_put(np.zeros((8, 4), dtype=np.float32), sharding)

  @jax.jit
  def fn(x):
    y = x + x
    return [y for _ in range(200)]

  jax.device_get(fn(inp))

  while state:
    [x._arrays for x in fn(inp)]


def batch_inplace_while(inplace_op, state):

  @jax.jit
  @jax.vmap
  def f(init_step, init_xs):

    def cond(carry):
      step, xs = carry
      return step < xs.size

    def body(carry):
      step, xs = carry
      if inplace_op == 'scatter':
        xs = xs.at[step].set(1)
      elif inplace_op == 'dynamic_update_slice':
        xs = lax.dynamic_update_index_in_dim(xs, 1., step, 0)
      else:
        assert False
      return step + 1, xs

    return lax.while_loop(cond, body, (init_step, init_xs))

  size = 100_000
  args = jnp.array([0]), jnp.zeros((1, size))
  jax.block_until_ready(f(*args))  # compile
  while state:
    jax.block_until_ready(f(*args))


google_benchmark.register(
    partial(batch_inplace_while, 'scatter'), name='batch_inplace_while_scatter')
google_benchmark.register(
    partial(batch_inplace_while, 'dynamic_update_slice'),
    name='batch_inplace_while_dynamic_update_slice')


@google_benchmark.register
def serial_dot_products(state):
  SIZE = 50

  @jax.jit
  @jax.vmap
  @jax.grad
  def f(x):
    out = 0
    for i in range(SIZE):
      y = x @ jnp.array([i, i + 1], dtype=jnp.float32)
      out = out + y * x[0]
    return out

  x = jax.random.normal(jax.random.PRNGKey(0), (2, 2))
  f(x).block_until_ready()  # compile
  while state:
    f(x).block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_names(['arg_lengths', 'num_args'])
@google_benchmark.option.args_product([[0, 1, 2, 5, 10, 100], [1, 2, 3]])
def safe_map(state):
  args = tuple(list(range(state.range(0))) for _ in range(state.range(1)))
  def f(*args): return tuple(args)
  while state:
    jax.util.safe_map(f, *args)

@google_benchmark.register
@google_benchmark.option.arg_names(['arg_lengths', 'num_args'])
@google_benchmark.option.args_product([[0, 1, 2, 5, 10, 100], [1, 2, 3]])
def safe_zip(state):
  args = tuple(list(range(state.range(0))) for _ in range(state.range(1)))
  while state:
    jax.util.safe_zip(*args)


@google_benchmark.register
def bench_make_array_from_callback_fully_replicated_sharding(state):
  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:8]).reshape((4, 2)), ('x', 'y'))
  shape = (8, 2)
  np_arr = np.arange(16).reshape(shape)
  s = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  while state:
    jax.make_array_from_callback(shape, s, np_arr.__getitem__)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_make_array_from_callback_sharded(state):
  global_mesh = create_mesh((4, 2), ('x', 'y'), state)
  input_shape = (8, 2)
  input_data = np.arange(math.prod(input_shape)).reshape(input_shape)

  def callback(index):
    return input_data[index]

  s = jax.NamedSharding(global_mesh, jax.sharding.PartitionSpec('x', 'y'))
  while state:
    jax.make_array_from_callback((8, 2), s, callback)

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def benchmark_lorentz63_cache_hits(state):
  @jax.jit
  def lorentz63(state, dt=0.01, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    x_t = sigma * (y - x)
    y_t = (rho - z) * x - y
    z_t = x * y - beta * z
    return jnp.array([x + x_t * dt, y + y_t * dt, z + z_t * dt])

  def training_step(initial_conditions, steps=1, unroll=False):
    def forward_sim(x0):
      if unroll:
        x = x0
        for _ in range(steps):
          x = lorentz63(x)
        return x
      else:
        return jax.lax.fori_loop(0, steps, lambda _, x: lorentz63(x), x0)

    def loss(x0):
      out = jax.vmap(jax.remat(forward_sim))(x0)
      return jnp.square(out).sum()

    return jax.value_and_grad(loss)(initial_conditions)

  x = jnp.ones((8, 3))
  while state:
    jax.make_jaxpr(lambda x: training_step(x, 100, unroll=True))(x)


@google_benchmark.register
def jit_add_chain(state):
  SIZE = 100

  @jax.jit
  def g(x, y):
    return lax.add(x, y)

  x = jax.random.normal(jax.random.PRNGKey(0), (2, 2))
  while state:
    @jax.jit
    def f(x):
      for i in range(SIZE):
        x = g(x, x)
      return x
    f(x).block_until_ready()


if __name__ == "__main__":
  google_benchmark.main()
