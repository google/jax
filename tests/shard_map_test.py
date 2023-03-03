# Copyright 2023 The JAX Authors.
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
import itertools as it
import math
import os
from types import SimpleNamespace
from typing import (Any, Sequence, Set, Iterable, Iterator, NamedTuple,
                    Callable, Optional, Tuple, List, Generator, TypeVar, Union)
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
from jax import lax
from jax.config import config
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.util import safe_zip, safe_map, partition_list, merge_lists
import jax.numpy as jnp

from jax.experimental.shard_map import shard_map

config.parse_flags_with_absl()

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# Helper for some tests.
def create_inputs(a_sharding, b_sharding):
  x, y, z = 2, 2, 2  # pylint: disable=invalid-name
  devices = np.array(jax.devices()[:x * y * z]).reshape((x, y, z))
  mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
  b, e, f = 8, 8, 8  # pylint: disable=invalid-name
  m1 = jax.device_put(
      jnp.arange(b * e).reshape((b, e)),
      jax.sharding.NamedSharding(mesh, a_sharding))
  m2 = jax.device_put(
      jnp.arange(e * f).reshape((e, f)),
      jax.sharding.NamedSharding(mesh, b_sharding))
  return mesh, m1, m2

# Run all tests with 8 CPU devices.
prev_xla_flags = None

# Run all tests with 8 CPU devices.
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

  if len(jax.devices()) < 8:
    raise unittest.SkipTest("tests require 8 devices")
  if not jax.config.jax_array:
    raise unittest.SkipTest("tests require jax_array")

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class ShardMapTest(jtu.JaxTestCase):

  def test_identity(self):
    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))
    assert a.device_buffers[0].shape == (4, 2)

    def identity(x):
      return x

    @jax.jit
    def fwd(a):
      c = shard_map(
          lambda x: x,
          mesh,
          in_specs=(P('z', ('x', 'y')),),
          out_specs=P('z', ('x', 'y')))(a)
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 2))

  def test_all_gather(self):
    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))
    assert a.device_buffers[0].shape == (4, 2)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', ('x', 'y')),), out_specs=P(None, ('x', 'y')))
    def fwd(a):
      return lax.all_gather(a, 'z', axis=0, tiled=True)

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (8, 2))

  def test_matmul_partial(self):
    raise unittest.SkipTest("invalid replication asserted by out_spec?")

    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))
    assert a.device_buffers[0].shape == (4, 4)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', 'y'), P('y', None)), out_specs=P('z', None))
    def fwd(a):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 8))

  def test_matmul_reduce_scatter(self):
    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))
    assert a.device_buffers[0].shape == (4, 4)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', 'y'), P('y', None)),
             out_specs=P(('z', 'y'), None))
    def fwd(a, b):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return lax.psum_scatter(c, 'y', scatter_dimension=0, tiled=True)

    c = fwd(a, b)
    self.assertEqual(c.device_buffers[0].shape, (2, 8))

  def test_collective_permute(self):
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P('x', None),),
             out_specs=P('x', None))
    def fwd(a):
      axis_size = lax.psum(1, 'x')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(a, 'x', perm=perm)

    c = fwd(a)
    self.assertAllClose(c[1, :], a[0, :])

  @jtu.skip_on_devices("cpu")  # all_to_all has a warning on cpu
  def test_all_to_all(self):
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P(None, 'x'))
    def fwd(a):
      return lax.all_to_all(a, 'x', split_axis=1, concat_axis=1, tiled=True)

    c = fwd(a)
    assert (c == jnp.reshape(a.T, (1, 64))).all()

  def test_eager_repr(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    s = None

    @partial(shard_map, mesh=mesh, in_specs=P('x', 'y'), out_specs=P('x', 'y'))
    def f(x):
      nonlocal s
      s = str(x)
      return x
    _ = f(np.arange(8 * 8.).reshape(8, 8))

    self.assertIsInstance(s, str)
    self.assertIn('at mesh coordinates', s)

  def test_jvp_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
    args = np.arange(4 * 4.).reshape(4, 4),
    jtu.check_grads(g, args, 2, ['fwd'])
    jtu.check_grads(jax.jit(g), args, 2, ['fwd'])

  def test_linearize_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
    x = np.arange(4 * 4.).reshape(4, 4)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_linearize_basic_repres(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jax.lax.sin(jax.lax.cos(x)), mesh,
                  in_specs=(P('x',),), out_specs=P('x',))
    x = np.arange(4.)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_linearize_basic_repres_jit(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x',),), out_specs=P('x',))
    x = np.arange(4.)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_replication_checker_eager(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = np.arange(8 * 8.).reshape(8, 8)

    def f(x):
      return 2 * x
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)

    with self.assertRaisesRegex(ValueError, 'statically inferred'):
      g(x)

    def f2(x):
      return jax.lax.psum(x, 'x')
    def g2(x):
      return shard_map(f2, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
    _ = g2(x)  # doesn't crash

  def test_replication_checker_jit(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = np.arange(8 * 8.).reshape(8, 8)

    def f(x):
      return 2 * x
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)

    with self.assertRaisesRegex(ValueError, 'statically inferred'):
      jax.jit(g)(x)

    def f2(x):
      return jax.lax.psum(x, 'x')
    def g2(x):
      return shard_map(f2, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
    _ = jax.jit(g2)(x)  # doesn't crash

  def test_process_env_traces(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))
    x = np.arange(8.)

    def g(x):
      y = (3. * x).sum()
      z = shard_map(lambda x: 2 * x * y, mesh,
                    in_specs=(P('x'),), out_specs=P('x'))(np.arange(8.))
      return z

    jtu.check_grads(g, (x,), modes=['fwd'], order=2)

  def test_eager_control_flow(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = jnp.arange(2 * 2.).reshape(2, 2)

    def f(x):
      y = jax.lax.psum(x, ('x', 'y'))
      if y < 0:
        return x
      else:
        return -x

    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))(x)
    y = g(x)
    self.assertAllClose(y, -x, check_dtypes=False)

  def test_outer_jit_detects_shard_map_mesh(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    f = shard_map(lambda x: x.reshape(1, *x.shape), mesh, P(), P('x'))
    _ = jax.jit(f)(jnp.array(2.0))  # doesnt crash

  def test_vmap_basic(self):
    if jax.config.jax_jit_pjit_api_merge:
      raise unittest.SkipTest("pjit batcher error")  # TODO(mattjj)

    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = jnp.arange(8 * 8.).reshape(8, 8)

    def g(x):
      return shard_map(lambda x: 2. * x, mesh,
                       in_specs=P('y'), out_specs=P('y'))(x)
    y = jax.vmap(g, axis_name='x')(x)
    self.assertAllClose(y, 2 * x, check_dtypes=False)

  def test_tree_prefix_error(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @partial(shard_map, mesh=mesh, in_specs=([P('x', 'y')],), out_specs=P('x', 'y'))
    def f(x):
      return x

    x = jnp.arange(8 * 8.).reshape(8, 8)
    with self.assertRaisesRegex(ValueError, r'shard_map in_specs\[0\]'):
       f([x, x])

  def test_rank_errors(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    def foo():
      return {'hi': [3.]}

    with self.assertRaisesRegex(ValueError, 'which has length 1'):
      shard_map(foo, mesh=mesh, in_specs=(), out_specs={'hi': P('x')})()

    with self.assertRaisesRegex(ValueError, 'which has length 1'):
      jax.jit(lambda: shard_map(foo, mesh=mesh,
                                in_specs=(), out_specs={'hi': P('x')})())()

    with self.assertRaisesRegex(ValueError, 'which has rank 0'):
      shard_map(foo, mesh=mesh, in_specs=({'hi': P('x')},), out_specs=())(
          {'hi': [jnp.array(3.)]})

  def test_reverse_mode_ad(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x',), P(None)), out_specs=P('x',))
    def f(x, y):
      return jnp.sin(x) + 3 + jnp.tan(2.) * jnp.cos(x) + y

    x = jnp.arange(8.) / 10.
    y = jnp.arange(4.) / 10.
    jtu.check_grads(f, (x, y), modes=['fwd', 'rev'], order=2)

  def test_post_process(self):
    # JVPTrace.post_process_shard_map and JaxprTrace.post_process_shard_map
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    def f(x):
      @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
      def g(y):
        return jnp.sin(y) * jnp.sin(x).sum()
      return g(jnp.arange(8.))

    x = jnp.arange(8.)
    _, f_lin = jax.linearize(f, x)
    y_dot = f_lin(x)

    y_dot_expected = jnp.sin(jnp.arange(8.)) * (jnp.cos(x) * x).sum()
    self.assertAllClose(y_dot, y_dot_expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu")
  def test_axis_index(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=P('x'))
    def f():
      return jax.lax.axis_index('x')[None]

    x = f()
    self.assertAllCLose(x, jnp.arange(4), check_dtypes=False)

  def test_remat_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))

    # check param updating is handled
    @jax.remat
    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f(x):
      return jnp.sin(x)

    x = jnp.arange(4.)
    g = jax.grad(lambda x: f(x).sum())(x)  # doesn't crash
    self.assertAllClose(g, jnp.cos(x), check_dtypes=False)

    # also check residuals are handled correctly
    @partial(jax.remat, policy=jax.checkpoint_policies.everything_saveable)
    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f2(x):
      return jnp.sin(x)

    g2 = jax.grad(lambda x: f2(x).sum())(x)  # doesn't crash
    self.assertAllClose(g2, jnp.cos(x), check_dtypes=False)

  def test_check_rep_false_doesnt_hit_rep_rules(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))

    prim = core.Primitive('prim')  # no rep rule here!
    prim.multiple_results = True
    prim.def_impl(lambda: [])
    prim.def_abstract_eval(lambda: [])

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=None, check_rep=True)
    def f():
      prim.bind()

    with self.assertRaises(NotImplementedError):
      f()
    with self.assertRaises(NotImplementedError):
      jax.jit(f)()

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=None, check_rep=False)
    def f2():
      prim.bind()

    f2()
    jax.jit(f2)()

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=None, check_rep=False)
    def f3():
      jax.jit(prim.bind)()

    f3()
    jax.jit(f3)()

  def test_vmap_spmd_axis_name(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f(x):
      return x

    x = jnp.arange(4 * 4).reshape(4, 4)
    jaxpr = jax.make_jaxpr(jax.vmap(f, spmd_axis_name='y'))(x).jaxpr
    e, = jaxpr.eqns
    self.assertIn('in_names', e.params)
    self.assertEqual(e.params['in_names'], ({0: ('y',), 1: ('x',)},))
    self.assertIn('out_names', e.params)
    self.assertEqual(e.params['out_names'], ({0: ('y',), 1: ('x',)},))

  def test_vmap_spmd_axis_name_pair(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P())
    def f(x):
      return x

    x = jnp.arange(4 * 4).reshape(4, 4)
    jaxpr = jax.make_jaxpr(jax.vmap(f, spmd_axis_name=('x', 'y')))(x).jaxpr
    e, = jaxpr.eqns
    self.assertIn('in_names', e.params)
    self.assertEqual(e.params['in_names'], ({0: ('x', 'y',)},))
    self.assertIn('out_names', e.params)
    self.assertEqual(e.params['out_names'], ({0: ('x', 'y',)},))


class FunSpec(NamedTuple):
  name: str
  num_inputs: int
  fun: Callable
  out_rep: Callable
  valid_types: Optional[Callable] = None

fun_specs = [
    FunSpec('id', 1, lambda x: x, lambda r: r),
    FunSpec('flip', 2, lambda x, y: (y, x), lambda r_x, r_y: (r_y, r_x)),
    FunSpec('transpose', 1, lambda x: x.T, lambda r: r),
    FunSpec('ravel', 1, lambda x: x.ravel(), lambda r: r),
    FunSpec(
        'dot', 2, jnp.dot, lambda r1, r2: r1 & r2,
        lambda x1, x2: (x1.shape and x2.shape and
                        x1.shape[-1] == x2.shape[-2 if x2.ndim > 1 else 0]),
             ),
    FunSpec(
        'dot_const', 1, lambda x: jnp.dot(x, np.ones(x.shape[-1:], x.dtype)),
        lambda r: r),
    FunSpec(
        'sin_dot_sin', 2,
        lambda x1, x2: jnp.sin(jnp.dot(jnp.sin(x1), x2)),
        lambda r1, r2: r1 & r2,
        lambda x1, x2: (x1.shape and x2.shape and
                        x1.shape[-1] == x2.shape[-2 if x2.ndim > 1 else 0])),
]

input_shapes = [
    jax.ShapeDtypeStruct(shape, jnp.dtype('float32'))
    # TODO(mattjj): 0 axis sizes lead to XLA sigfpe, file bug!
    for k in range(1, 4) for shape in it.permutations(range(1, 4), k)
    if not shape or len(set(shape)) > 1  # skip all-equal shapes, boring!
]

mesh_shapes = [
    (1,),
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 4),
    (4, 2),
]

# Reference implementation of shard_map.

ShapeDtypeDuck = Any  # has shape and dtype attributes
Specs = Any  # pytree of PartitionSpec

def shmap_reference(
    body_in_types: Sequence[ShapeDtypeDuck],
    body_out_types: Sequence[ShapeDtypeDuck],
    out_types: Sequence[ShapeDtypeDuck],
    f: Callable, mesh: Mesh, in_specs: Specs, out_specs: Specs
  ) -> Callable:
  def f_shmapped(*args):
    outs = jax.tree_map(lambda y: jnp.zeros(y.shape, y.dtype), out_types)
    getters = [make_indexer(mesh, s, x) for s, x in zip(in_specs, args)]
    putters = jax.tree_map(partial(make_indexer, mesh), out_specs, outs)
    for idx in it.product(*map(range, mesh.shape.values())):
      args_shards = [x[indexer(idx)] for x, indexer in zip(args, getters)]
      assert all(x.shape == r.shape for x, r in zip(args_shards, body_in_types))
      out_shards = f(*args_shards)
      assert jax.tree_util.tree_all(jax.tree_map(lambda y, r: y.shape == r.shape,
                                                 out_shards, body_out_types))
      outs = jax.tree_map(lambda y, out, indexer: out.at[indexer(idx)].set(y),
                          out_shards, outs, putters)
    return outs
  return f_shmapped

def make_indexer(mesh: Mesh, spec: P, x: Any
                 ) -> Callable[[Tuple[int, ...]], Tuple[slice, ...]]:
  block_shape = [d // math.prod(mesh.shape[ax] for ax in (elt or ()))
                 for d, elt in zip(x.shape, spec)]
  def indexer(idx):
    starts = [0 if el is None else
              idx[list(mesh.shape).index(el)] if type(el) is not tuple else
              sum(idx[list(mesh.shape).index(el[i])]
                  * math.prod(mesh.shape[e] for e in el[i+1:]) for i in range(len(el)))
              for el in spec]
    return tuple(slice(start * size, (start + 1) * size)
                 for start, size in zip(starts, block_shape))
  return indexer


# The code below is similar to named_cases_from_sampler in test_util.py, but it
# uses generators instead of passing a "select" function around.

# To sample test cases efficiently, we construct a generator which yields to the
# caller to choose one of an iterable's options. That is, we can read 'yield' in
# this code as 'choose one'. To call functions which themselves need to make
# choices, we use 'yield from'. That is, we can read 'yield from' in this code
# as 'call this choice-making function'.
Option = Any
CaseSpec = Tuple  # first element is a string test name
Chooser = Generator[Iterable[Option], Option, CaseSpec]

def sample_shmap() -> Chooser:
  spec = yield fun_specs
  mesh_shape = yield mesh_shapes
  axis_names = ('i', 'j', 'k', 'l')[:len(mesh_shape)]
  mesh = SimpleNamespace(shape=dict(zip(axis_names, mesh_shape)),
                         axis_names=axis_names)
  in_types = (tys for tys in it.product(input_shapes, repeat=spec.num_inputs)
              if not spec.valid_types or spec.valid_types(*tys))
  body_in_types = yield in_types
  body_out_types = jax.eval_shape(spec.fun, *body_in_types)
  in_types, in_specs = yield from make_in_specs(mesh, body_in_types)
  args = [np.arange(ty.size, dtype=ty.dtype).reshape(ty.shape) / ty.size
          for ty in in_types]
  out_reps = spec.out_rep(*map(partial(unmentioned, mesh), in_specs))
  out_specs = yield from make_out_specs(mesh, body_out_types, out_reps)
  out_types = jax.tree_map(partial(dilate, mesh), out_specs, body_out_types)
  ref = partial(shmap_reference, body_in_types, body_out_types, out_types)
  in_str = '(' + ','.join(jax.core.ShapedArray(t.shape, t.dtype).str_short()
                          for t in in_types) + ')'
  name = f'{spec.name}_{mesh.shape}_{in_specs}_{out_specs}_{in_str}'
  return name, spec.fun, mesh.shape, in_specs, out_specs, args, ref

def unmentioned(mesh: Mesh, pspec: P) -> Set[core.AxisName]:
  return set(mesh.axis_names) - {n for ns in pspec if ns is not None
                                 for n in (ns if type(ns) is tuple else [ns])}


# To drive the sampler, we have `sample` function which just runs a loop.
def sample(num: int, make_gen: Callable[[], Chooser]) -> Iterator[CaseSpec]:
  rng = np.random.RandomState(0)
  seen: Set[str] = set()
  while len(seen) < num:
    name, *case = sample_one(rng, make_gen())
    if name not in seen:
      seen.add(name)
      yield name, *case

# To sample one test spec, we run the generator, getting back sequences of
# options from it and sending in our choices from those options until finally a
# test case spec is produced.
def sample_one(rng: np.random.RandomState, gen: Chooser) -> CaseSpec:
  lst = list(next(gen))
  try:
    while True:
      choice = lst[rng.randint(len(lst))]
      lst = list(gen.send(choice))
  except StopIteration as e:
    return e.value

# Next are some choice-making functions for shard_map test specifications.

MeshDuck = Any  # same attributes as a Mesh

def make_in_specs(mesh: MeshDuck, in_types: Sequence[ShapeDtypeDuck]
                  ) -> Chooser:
  pairs = []
  for ty in in_types:
    pair = yield from make_in_spec(mesh, ty)
    pairs.append(pair)
  return tuple(zip(*pairs))

def make_in_spec(mesh: Mesh, in_type_base: ShapeDtypeDuck) -> Chooser:
  assert len(list(powerset(mesh.shape)))
  subset = yield powerset(mesh.shape)
  elts = yield partitions(subset, len(in_type_base.shape))
  partition_spec = P(*(tuple(e) if e else None for e in elts))
  new_type = dilate(mesh, partition_spec, in_type_base)
  return new_type, partition_spec

def dilate(mesh: Mesh, spec: P, shape: ShapeDtypeDuck) -> ShapeDtypeDuck:
  new_shape = tuple(d * math.prod(mesh.shape[ax] for ax in (elt or ()))
                    for d, elt in zip(shape.shape, spec))
  return jax.ShapeDtypeStruct(new_shape, shape.dtype)

def make_out_specs(
    mesh: MeshDuck, out_types: Union[ShapeDtypeDuck, Sequence[ShapeDtypeDuck]],
    out_reps: Union[Set[core.AxisName], Sequence[Set[core.AxisName]]]
  ) -> Chooser:
  if type(out_types) is not tuple:
    out_spec = yield from make_out_spec(mesh, out_types, out_reps)  # type: ignore
    return out_spec
  else:
    out_specs = []
    for ty, rep in zip(out_types, out_reps):
      out_spec = yield from make_out_spec(mesh, ty, rep)  # type: ignore
      out_specs.append(out_spec)
    return tuple(out_specs)

def make_out_spec(
    mesh: Mesh, out_type: ShapeDtypeDuck, out_rep: Set[core.AxisName]
  ) -> Chooser:
  subset = yield (s for s in powerset(mesh.shape)
                  if out_rep | set(s) == set(mesh.shape))
  elts = yield partitions(subset, len(out_type.shape))
  return P(*(tuple(e) if e else None for e in elts))

# Combinatorial helper functions

T = TypeVar('T')
def partitions(s: Sequence[T], k: int) -> Iterator[List[List[T]]]:
  for indices in it.product(range(k), repeat=len(s)):
    outs: List[List[T]] = [[] for _ in range(k)]
    for i, elt in zip(indices, s):
      outs[i].append(elt)
    yield outs

def powerset(s: Iterable[T]) -> Iterator[Sequence[T]]:
  s = list(s)
  return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))


class ShardMapSystematicTest(jtu.JaxTestCase):

  @staticmethod
  def make_mesh(mesh_shape):
    return jtu.create_global_mesh(tuple(mesh_shape.values()), tuple(mesh_shape))

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases, sample_shmap))
  def test_eager_against_ref(self, fun, mesh, in_specs, out_specs, args, ref):
    mesh = self.make_mesh(mesh)
    out = shard_map(fun, mesh, in_specs, out_specs)(*args)
    expected = ref(fun, mesh, in_specs, out_specs)(*args)
    self.assertAllClose(expected, out, check_dtypes=False)

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases, sample_shmap))
  def test_jit_against_ref(self, fun, mesh, in_specs, out_specs, args, ref):
    mesh = self.make_mesh(mesh)
    out = jax.jit(shard_map(fun, mesh, in_specs, out_specs))(*args)
    expected = ref(fun, mesh, in_specs, out_specs)(*args)
    self.assertAllClose(expected, out, check_dtypes=False)

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases, sample_shmap))
  @jax.default_matmul_precision("float32")
  def test_grads(self, fun, mesh, in_specs, out_specs, args, _):
    raise unittest.SkipTest("internal xla failures")  # TODO(b/269660532)
    mesh = self.make_mesh(mesh)
    f = jax.jit(shard_map(fun, mesh, in_specs, out_specs))
    jtu.check_grads(f, args, order=2, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases, sample_shmap))
  @jax.default_matmul_precision("float32")
  def test_grads_closure(self, fun, mesh, in_specs, out_specs, args, _):
    raise unittest.SkipTest("internal xla failures")  # TODO(b/269660532)
    mesh = self.make_mesh(mesh)
    no_sharding = [all(elt is None for elt in spec) for spec in in_specs]
    args, closed_over_args = partition_list(no_sharding, args)
    in_specs, _ = partition_list(no_sharding, in_specs)
    def f(x, *closed_over_args):
      @jax.jit
      @partial(shard_map, mesh=mesh, in_specs=(*in_specs,), out_specs=out_specs)
      def g(*args):
        args = [x * arg for arg in args]
        args = merge_lists(no_sharding, args, closed_over_args)
        return fun(*args)
      return g(*args)
    jtu.check_grads(f, (0.2, *closed_over_args), order=2, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
