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

from collections.abc import Generator, Iterator
import contextlib
import functools
import itertools as it
from itertools import product, permutations
import math
import re
from unittest import SkipTest
from typing import Union

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax._src import test_util as jtu
from jax import vmap
from jax import lax
from jax.ad_checkpoint import checkpoint
from jax.errors import JAXTypeError
from jax.experimental.pjit import pjit
from jax.interpreters import batching
from jax.sharding import PartitionSpec as P
from jax._src import array
from jax._src import core
from jax._src import maps
from jax._src.maps import xmap, serial_loop, SerialLoop
from jax._src import xla_bridge
from jax._src.core import NamedShape
from jax._src.lax import parallel as lax_parallel
from jax._src.lax.parallel import pgather
from jax._src.lib import xla_client
from jax._src.nn import initializers as nn_initializers
from jax._src.sharding_impls import NamedSharding
from jax._src.util import unzip2

jax.config.parse_flags_with_absl()

# Run all tests with 8 CPU devices.
_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(8))

def tearDownModule():
  _exit_stack.close()


def create_array(global_shape, global_mesh, mesh_axes, global_data=None):
  if global_data is None:
    global_data = np.arange(
        math.prod(global_shape), dtype=np.float32).reshape(global_shape)

  sharding = NamedSharding(global_mesh, mesh_axes)

  return array.make_array_from_callback(
      global_shape, sharding, lambda idx: global_data[idx]), global_data


# -------------------- Itertools helpers --------------------

def partitions(s, k):
  for indices in product(range(k), repeat=len(s)):
    outs = [[] for _ in range(k)]
    for i, elt in zip(indices, s):
      outs[i].append(elt)
    yield outs

def powerset(s):
  s = list(s)
  return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

# -------------------- vmap test helpers --------------------

ensure_bdim_p = core.Primitive('ensure_bdim')
ensure_bdim_p.def_abstract_eval(lambda x, **kwargs: core.raise_to_shaped(x))
def _ensure_bdim_batcher(axis_size, frame_name, main_type, vals_in, dims_in, axis_name, bdim):
  v, = vals_in
  d, = dims_in
  assert d is not batching.not_mapped
  return jnp.moveaxis(v, d, bdim), bdim
batching.axis_primitive_batchers[ensure_bdim_p] = _ensure_bdim_batcher
batching.primitive_batchers[ensure_bdim_p] = lambda v, d: (v[0], d[0])
core.axis_substitution_rules[ensure_bdim_p] = partial(
    lax_parallel._subst_all_names_in_param, 'axis_name')

def ensure_bdim(x, axis_name, bdim):
  return ensure_bdim_p.bind(x, axis_name=(axis_name,), bdim=bdim)

# When we use the SPMD lowering, we vmap the xmap body to make all named
# axes positional again. This lowering can introduce constants, which we
# have to handle properly in the lowering rule.
constant_introducing_p = core.Primitive('introduce_constant')
constant_introducing_p.def_abstract_eval(lambda x, **_: core.raise_to_shaped(x))
def _constant_introducing_batcher(_1, _2, _3, xs, ds, axis_name):
  (x,), (d,) = xs, ds
  # Introduce a constant
  return (x + np.arange(x.size, dtype=x.dtype).reshape(x.shape)), d
batching.axis_primitive_batchers[constant_introducing_p] = _constant_introducing_batcher
core.axis_substitution_rules[constant_introducing_p] = partial(
  lax_parallel._subst_all_names_in_param, 'axis_name')

# -------------------- Axis resources generation --------------------

AxisResources = dict[str, Union[str, tuple[str, ...]]]

def schedules(sizes: dict[str, int]
              ) -> Generator[tuple[AxisResources, jtu.MeshSpec], None, None]:
  """Test utility generating xmap parallel schedules from logical names & sizes.

  Args:
    sizes: dict mapping logical axis name to its corresponding size.

  Returns:
    A generator producing finitely many values, where each value is a pair in
    which the first element is a value suitable for xmap's axis_resources
    argument and the second element is a list of pairs with the first element
    representing a generated physical mesh axis name and the second element
    representing a corresponding generated mesh axis size. The generated mesh
    names/sizes can be used to define a physical mesh in tests.

  This function doesn't generate schedules which map distinct logical axis names
  to the same parallel resource name. It only generates parallel resources; the
  rest are implicitly left for vectorization. Parallel resource names are
  generated by prepending an 'r', 'r1', or 'r2' to the corresponding logical
  name.

  Examples:
    >>> for sched in schedules({'i': 2, 'j': 4}):
    ...   print(sched)
    ({}, [])
    ({'i': 'ri'}, [('ri', 1)])
    ({'i': 'ri'}, [('ri', 2)])
    ({'i': ('r1i', 'r2i')}, [('r1i', 1), ('r2i', 1)])
    ({'i': ('r1i', 'r2i')}, [('r1i', 1), ('r2i', 2)])
    ({'i': ('r1i', 'r2i')}, [('r1i', 2), ('r2i', 1)])
    ({'j': 'rj'}, [('rj', 1)])
    ({'j': 'rj'}, [('rj', 2)])
    ({'j': 'rj'}, [('rj', 4)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 1), ('r2j', 1)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 1), ('r2j', 2)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 1), ('r2j', 4)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 2), ('r2j', 1)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 2), ('r2j', 2)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 4), ('r2j', 1)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 1), ('rj', 1)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 1), ('rj', 2)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 1), ('rj', 4)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 2), ('rj', 1)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 2), ('rj', 2)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 2), ('rj', 4)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 1), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 1), ('r2j', 2)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 1), ('r2j', 4)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 2), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 2), ('r2j', 2)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 4), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 1), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 1), ('r2j', 2)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 1), ('r2j', 4)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 2), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 2), ('r2j', 2)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 4), ('r2j', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 1), ('r1i', 1), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 1), ('r1i', 1), ('r2i', 2)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 1), ('r1i', 2), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 2), ('r1i', 1), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 2), ('r1i', 1), ('r2i', 2)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 2), ('r1i', 2), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 4), ('r1i', 1), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 4), ('r1i', 1), ('r2i', 2)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 4), ('r1i', 2), ('r2i', 1)])
  """
  def divisors(n: int) -> list[int]:
    return [m for m in range(1, n + 1) if not n % m]

  def divisors2(n: int) -> Iterator[tuple[int, int]]:
    for k1 in divisors(n):
      for k2 in divisors(n // k1):
        yield (k1, k2)

  # choose a subset of logical axis names to map to parallel resources
  for names in powerset(sizes):
    # partition that set of logical axis names into two subsets: one subset to
    # map to one parallel resource axis and a second subset to map to two
    # parallel resource axes.
    for names1, names2 in partitions(names, 2):
      # to avoid generating too many complex cases, we skip generating cases
      # where more than one logical axis name is to be mapped to two parallel
      # resource axes. comment out this line to generate more complex tests.
      if len(names2) > 1: continue
      # make up parallel resource axis names for each logical axis
      axis_resources1 = ((name, 'r' + name) for name in names1)
      axis_resources2 = ((name, ('r1' + name, 'r2' + name)) for name in names2)
      axis_resources = dict(it.chain(axis_resources1, axis_resources2))
      # make up sizes for each resource axis, where the size must divide the
      # corresponding logical axis
      for mesh_sizes1 in product(*(divisors(sizes[n]) for n in names1)):
        for mesh_sizes2 in product(*(divisors2(sizes[n]) for n in names2)):
          mesh_data1 = (('r' + name, size) for name, size in zip(names1, mesh_sizes1))
          mesh_data2 = (pair for name, (size1, size2) in zip(names2, mesh_sizes2)
                        for pair in [('r1' + name, size1), ('r2' + name, size2)])
          mesh_data = list(it.chain(mesh_data1, mesh_data2))
          yield axis_resources, mesh_data


@jtu.pytest_mark_if_available('multiaccelerator')
@jtu.with_config(jax_legacy_prng_key="allow")
class XMapTestCase(jtu.BufferDonationTestCase):
  pass


# A mixin that enables SPMD lowering tests
class SPMDTestMixin:
  def setUp(self):
    super().setUp()
    self.spmd_lowering = maps.SPMD_LOWERING.value
    jax.config.update('experimental_xmap_spmd_lowering', True)

  def tearDown(self):
    jax.config.update('experimental_xmap_spmd_lowering', self.spmd_lowering)


class ManualSPMDTestMixin:
  def setUp(self):
    if not hasattr(xla_client.OpSharding.Type, "MANUAL"):
      raise SkipTest
    super().setUp()
    self.spmd_lowering = maps.SPMD_LOWERING.value
    self.spmd_manual_lowering = maps.SPMD_LOWERING_MANUAL.value
    jax.config.update('experimental_xmap_spmd_lowering', True)
    jax.config.update('experimental_xmap_spmd_lowering_manual', True)

  def tearDown(self):
    jax.config.update('experimental_xmap_spmd_lowering', self.spmd_lowering)
    jax.config.update('experimental_xmap_spmd_lowering_manual', self.spmd_manual_lowering)


@jtu.pytest_mark_if_available('multiaccelerator')
class XMapTest(XMapTestCase):

  def testBasic(self):
    local_devices = list(jax.local_devices())
    if len(local_devices) < 4:
      raise SkipTest("Test requires at least 4 local devices")
    def f(a, b):
      return a * 2, b * 4
    devices = np.array(local_devices[:4]).reshape((2, 2))
    with jax.sharding.Mesh(devices, ('x', 'y')):
      fm = xmap(f,
                in_axes=({0: 'a', 1: 'b'}, ['c', ...]),
                out_axes=({0: 'a', 1: 'b'}, ['c', ...]),
                axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})
      ashape = (16, 8, 5)
      a = jnp.arange(math.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = jnp.arange(math.prod(bshape)).reshape(bshape)
      c, d = fm(a, b)
      self.assertAllClose(c, a * 2)
      self.assertAllClose(d, b * 4)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testCollectiveReduce(self):
    fm = xmap(lambda a, b: (lax.psum(a * 2, 'a'), b * 4),
              in_axes=(['a', 'b', ...], {0: 'c'}),
              out_axes=(['b', ...], {0: 'c'}),
              axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})
    ashape = (16, 8, 5)
    a = jnp.arange(math.prod(ashape)).reshape(ashape)
    bshape = (2, 7)
    b = jnp.arange(math.prod(bshape)).reshape(bshape)
    c, d = fm(a, b)
    self.assertAllClose(c, (a * 2).sum(0))
    self.assertAllClose(d, b * 4)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testCollectivePermute2D(self):
    perm = np.array([3, 1, 2, 0])
    x = jnp.arange(4).reshape((2, 2))
    result = xmap(lambda x: lax.pshuffle(x, ('i', 'j'), perm),
                  in_axes=['i', 'j', ...],
                  out_axes=['i', 'j', ...],
                  axis_resources={'i': 'x', 'j': 'y'})(x).reshape((-1,))
    self.assertAllClose(result, perm, check_dtypes=False)

  def testCollectivePermute1D(self):
    perm = np.array([3, 1, 2, 0])
    x = jnp.arange(4)
    result = xmap(lambda x: lax.pshuffle(x, 'i', perm),
                  in_axes=['i', ...],
                  out_axes=['i', ...])(x)
    self.assertAllClose(result, perm, check_dtypes=False)

  def testCollectiveAllGather(self):
    x = jnp.arange(4, dtype='int32')
    result = xmap(lambda x: lax.all_gather(x, 'i') + lax.axis_index('i'),
                  in_axes=['i', ...], out_axes=['i', ...])(x)
    self.assertAllClose(result, x[jnp.newaxis] + x[jnp.newaxis].T)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testOneLogicalTwoMeshAxesBasic(self):
    def f(v):
      return lax.psum(v * 2, 'a'), v * 4
    fm = xmap(f, in_axes=['a', ...], out_axes=({}, {1: 'a'}),
              axis_resources={'a': ('x', 'y')})
    vshape = (4, 5)
    v = jnp.arange(math.prod(vshape)).reshape(vshape)
    ans, ans2 = fm(v)
    self.assertAllClose(ans, (v * 2).sum(0))
    self.assertAllClose(ans2, v.T * 4)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testOneLogicalTwoMeshAxesSharding(self):
    def f(v):
      return v * 4
    fxy = xmap(f, in_axes=['a', ...], out_axes={1: 'a'},
               axis_resources={'a': ('x', 'y')})
    fyx = xmap(f, in_axes=['a', ...], out_axes={1: 'a'},
               axis_resources={'a': ('y', 'x')})
    vshape = (4, 5)
    v = jnp.arange(math.prod(vshape)).reshape(vshape)
    zxy = fxy(v)
    zxy_op_sharding = zxy.sharding._to_xla_hlo_sharding(zxy.ndim)
    self.assertListEqual(zxy_op_sharding.tile_assignment_dimensions(), [1, 4])
    self.assertListEqual(zxy_op_sharding.tile_assignment_devices(), [0, 1, 2, 3])
    zyx = fyx(v)
    zyx_op_sharding = zyx.sharding._to_xla_hlo_sharding(zyx.ndim)
    self.assertListEqual(zyx_op_sharding.tile_assignment_dimensions(), [1, 4])
    self.assertListEqual(zyx_op_sharding.tile_assignment_devices(), [0, 2, 1, 3])

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testSkipFirstMeshDim(self):
    def run(axis_resources):
      return xmap(lambda x: x * 2, in_axes=['i', ...], out_axes=['i', ...],
                  axis_resources=axis_resources)(jnp.ones((4,)))
    self.assertAllClose(run({'i': 'x'}), run({'i': 'y'}))

  def testCaching(self):
    def f(x):
      assert python_should_be_executing
      return x * 2
    devices = np.array(jax.local_devices()[:2])
    if devices.size < 2:
      raise SkipTest("Test requires 2 devices")
    x = np.arange(8).reshape((2, 2, 2))
    with jax.sharding.Mesh(devices, ('x',)):
      python_should_be_executing = True
      xmap(f, in_axes=['a', ...], out_axes=['a', ...],
           axis_resources={'a': 'x'})(x)
      python_should_be_executing = False
      xmap(f, in_axes=['a', ...], out_axes=['a', ...],
           axis_resources={'a': 'x'})(x)
    with jax.sharding.Mesh(devices, ('x',)):
      python_should_be_executing = False
      xmap(f, in_axes=['a', ...], out_axes=['a', ...],
           axis_resources={'a': 'x'})(x)

  def testNoTracerLeak(self):
    self.skipTest('Does not work with Array because of ShardingContext '
                  'being used in xmap because of jit. Removing that '
                  'restriction makes the test pass but that should be done '
                  'in a separate CL.')
    @jax.jit
    def xmap_linearize(xs):
      eye = jnp.eye(xs.shape[0], dtype=jnp.float32)
      primal, grad_f = jax.linearize(jnp.sin, xs)
      return maps.xmap(
          grad_f,
          in_axes=['i', ...],
          out_axes=['i', ...],
          axis_resources={'i': maps.SerialLoop(1)})(eye)
    xs = jnp.arange(1, 4, step=1).astype(jnp.float32)
    xmap_linearize(xs)  # Doesn't raise a tracer leak error

  @parameterized.named_parameters(
    {"testcase_name": name, "mesh": mesh, "axis_resources": axis_resources}
    for name, mesh, axis_resources in (
      ('OneToOne', (('x', 2), ('y', 2)), (('a', 'y'), ('b', 'x'))),
      ('Multiple', (('x', 2), ('y', 2), ('z', 2)), (('a', 'y'), ('b', ('x', 'z')))),
    ))
  @jtu.with_mesh_from_kwargs
  @jax.numpy_dtype_promotion('standard')
  def testNestedMesh(self, mesh, axis_resources):
    @partial(xmap, in_axes={1: 'a'}, out_axes=({0: 'a'}, {}),
              axis_resources=dict([axis_resources[0]]))
    def f(x):
      y = x * 2
      @partial(xmap, in_axes={0: 'b'}, out_axes=({1: 'b'}, {}),
               axis_resources=dict([axis_resources[1]]))
      def h(y):
        # Multiply by a constant array to better exercise the partial_eval rule
        return jnp.sin(y) * np.arange(y.size, dtype=float), lax.psum(y, ('a', 'b'))
      return h(y)

    xshape = (4, 2, 5)
    x = jnp.arange(math.prod(xshape), dtype=float).reshape(xshape)
    y = f(x)
    self.assertAllClose(
        y, ((jnp.sin(x * 2) *
             np.arange(xshape[-1], dtype=float)[None, None]).transpose(
                 (1, 2, 0)), (x * 2).sum((0, 1))))

    y_op_sharding = y[0].sharding._to_xla_hlo_sharding(y[0].ndim)
    m_size = math.prod([2] + [2] * (len(mesh) - 2))
    self.assertListEqual(y_op_sharding.tile_assignment_dimensions(),
                         [2, 1, 1, m_size])
    if maps.SPMD_LOWERING.value:
      hlo = f.lower(x).compiler_ir(dialect="hlo").as_hlo_text()
      # Make sure that there are non-partial sharding specs in the HLO
      self.assertRegex(
          hlo, r'sharding={devices=\[[0-9,]+\]<=\[[0-9,]+\](T\([0-9,]+\))?}'
      )

  @jtu.with_and_without_mesh
  def testMultipleCalls(self, mesh, axis_resources):
    def f(x, y):
      assert x.shape == y.shape == (3, 5)
      return jnp.tensordot(x, y, axes=([1], [1]))

    f_mapped = xmap(f,
                    in_axes=(['i', ...], ['j', ...]),
                    out_axes=['i', 'j', ...],
                    axis_resources=dict(axis_resources))
    x = jnp.arange(30).reshape(2, 3, 5)
    expected = jnp.einsum('imk,jnk->ijmn', x, x)
    for i in range(10):
      self.assertAllClose(f_mapped(x, x), expected)

  @jtu.with_and_without_mesh
  @jtu.device_supports_buffer_donation()  # In/out aliasing not supported on CPU
  def testBufferDonation(self, mesh, axis_resources):
    shard = lambda x: x
    if axis_resources:
      shard = xmap(lambda x: x, in_axes=['i', ...], out_axes=['i', ...],
                   axis_resources=dict(axis_resources))
    f = xmap(lambda x, y: x + y * 4,
             in_axes=['i', ...], out_axes=['i', ...],
             axis_resources=dict(axis_resources),
             donate_argnums=0)
    # The multiplications below disable some optimizations that prevent reuse
    x = shard(jnp.zeros((2, 5)) * 4)
    y = shard(jnp.ones((2, 5)) * 2)
    f(x, y)
    self.assertNotDeleted(y)
    self.assertDeleted(x)

  @jtu.device_supports_buffer_donation()  # In/out aliasing not supported on CPU
  @jtu.with_mesh([('x', 2)])
  @jtu.ignore_warning(category=UserWarning,  # SPMD test generates warning.
                      message="Some donated buffers were not usable*")
  def testBufferDonationNamedShape(self):
    axis_resources = {'i': 'x'}
    # named in_aval, unnamed out_aval
    f = xmap(lambda _: jnp.ones((2, 5)),
             in_axes=['i', ...], out_axes=[...],
             axis_resources=axis_resources,
             donate_argnums=0)
    shard = xmap(lambda x: x, in_axes=['i', ...], out_axes=['i', ...],
                 axis_resources=dict(axis_resources))
    x = shard(jnp.zeros((4, 5)))
    f(x)
    self.assertDeleted(x)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testConstantsInLowering(self):
    h = xmap(partial(constant_introducing_p.bind, axis_name='i'),
             in_axes=['i'], out_axes=['i'], axis_resources={'i': 'x'})
    f = xmap(h, in_axes=['j', ...], out_axes=['j', ...], axis_resources={'j': 'y'})

    yp = 1 + jnp.arange(10, dtype=np.float32)
    self.assertAllClose(
      f(jnp.ones((2, 20), dtype=np.float32)),
      jnp.broadcast_to(jnp.concatenate([yp, yp]), (2, 20)))

  def testControlFlow(self):
    x = jnp.arange(5)
    xmap(lambda x: lax.fori_loop(0, 10, lambda _, x: lax.psum(x, 'i'), x),
         in_axes=['i', ...], out_axes=['i', ...])(x)

  @jtu.with_and_without_mesh
  def testAxisSizes(self, mesh, axis_resources):
    result = xmap(lambda: lax.axis_index('i'),
                  in_axes=(), out_axes=['i', ...],
                  axis_sizes={'i': 6},
                  axis_resources=dict(axis_resources))()
    self.assertAllClose(result, jnp.arange(6, dtype=result.dtype))

  def testCollectiveOverNoName(self):
    result = xmap(lambda: lax.psum(jnp.array(2) ** 2, 'i'),
                  in_axes={}, out_axes={}, axis_sizes={'i': 4})()
    self.assertEqual(result, 16)

  def VmapOfXmapCases(s):
    xmap_in_axes = ([{}] +
                    [{i: 'x'} for i in range(3)] +
                    [{i: 'x', j: 'y'} for i in range(4) for j in range(4) if i != j])
    for xmap_dim_x, xmap_dim_y in s(product(xmap_in_axes, repeat=2)):
      xmap_axes = sorted(set(xmap_dim_x.values()) | set(xmap_dim_y.values()))
      num_axes = len(xmap_axes)
      if xmap_axes is None:
        continue
      xmap_out_axes = [dict(zip(dims, xmap_axes))
                       for dims in permutations(range(2 + num_axes), num_axes)]
      for xmap_dim_z in s(xmap_out_axes):
        for vmap_dim_x in s([*range(2 + len(xmap_dim_x)), None]):
          for vmap_dim_y in s([*range(2 + len(xmap_dim_y)), None]):
            if vmap_dim_x is None and vmap_dim_y is None:
              continue
            for vmap_dim_result in s(range(3)):
              for vmap_dim_z in s(range(2 + len(xmap_axes))):
                for vmap_as_xmap in s([False, True]):
                  yield {"testcase_name":
                             f"_xin={(sorted(xmap_dim_x.items()), sorted(xmap_dim_y.items()))}_"
                             f"xout={sorted(xmap_dim_z.items())}_vin={(vmap_dim_x, vmap_dim_y)}_"
                             f"vout={vmap_dim_z}_vresult={vmap_dim_result}_{vmap_as_xmap=}",
                         "xmap_in_axes": (xmap_dim_x, xmap_dim_y),
                         "xmap_out_axes": xmap_dim_z,
                         "vmap_in_axes": (vmap_dim_x, vmap_dim_y),
                         "vmap_out_axes": vmap_dim_z,
                         "vmap_result_axis": vmap_dim_result,
                         "vmap_as_xmap": vmap_as_xmap}

  @parameterized.named_parameters(jtu.named_cases_from_sampler(VmapOfXmapCases))
  @jax.default_matmul_precision("float32")
  def testNestedMap(self,
                    xmap_in_axes, xmap_out_axes,
                    vmap_in_axes, vmap_out_axes, vmap_result_axis,
                    vmap_as_xmap):
    """Test various vmap(xmap) and xmap(xmap) combinations.

    The outer map always introduces a single dimension, the inner map introduces one or two.
    """
    (xin_x, xin_y) = xmap_in_axes
    (vin_x, vin_y) = vmap_in_axes
    vmap_size = 7
    xmap_sizes = {'x': 11, 'y': 13}

    xshape = [2, 3]
    yshape = [3, 5]
    zshape = [2, 5]
    xind = ['n', 'k']
    yind = ['k', 'm']
    zind = ['n', 'm']
    f = lambda x, y: ensure_bdim(jnp.einsum('nk,km->nm', x, y), 'v', vmap_result_axis)

    for pos, name in sorted(xin_x.items()):
      xshape.insert(pos, xmap_sizes[name])
      xind.insert(pos, name)
    for pos, name in sorted(xin_y.items()):
      yshape.insert(pos, xmap_sizes[name])
      yind.insert(pos, name)
    for pos, name in sorted(xmap_out_axes.items()):
      zshape.insert(pos, xmap_sizes[name])
      zind.insert(pos, name)

    if vin_x is not None:
      xshape.insert(vin_x, vmap_size)
      xind.insert(vin_x, 'v')
    if vin_y is not None:
      yshape.insert(vin_y, vmap_size)
      yind.insert(vin_y, 'v')
    zshape.insert(vmap_out_axes, vmap_size)
    zind.insert(vmap_out_axes, 'v')

    if vmap_as_xmap:
      do_vmap = partial(xmap,
                        in_axes=({vin_x: 'v'} if vin_x is not None else {},
                                 {vin_y: 'v'} if vin_y is not None else {}),
                        out_axes={vmap_out_axes: 'v'})
    else:
      do_vmap = partial(vmap, in_axes=vmap_in_axes, out_axes=vmap_out_axes, axis_name='v')

    fm = do_vmap(xmap(f, in_axes=xmap_in_axes, out_axes=xmap_out_axes))
    fref = partial(jnp.einsum, f"{''.join(xind)},{''.join(yind)}->{''.join(zind)}")

    rng = self.rng()
    x = rng.randn(*xshape)
    y = rng.randn(*yshape)
    self.assertAllClose(fm(x, y), fref(x, y), atol={np.float64: 1e-14})

  def testBatchingPostProcess(self):
    x = jnp.arange(10).reshape(5, 2)
    f = jax.vmap(lambda y: xmap(lambda x: x + y, in_axes=['i', ...], out_axes=['i', ...])(x))
    ref = jax.vmap(lambda y: jax.vmap(lambda x: x + y)(x))
    self.assertAllClose(f(x * 2), ref(x * 2))

  def testAutodiffBroadcast(self):
    raise SkipTest("reduce_axes on vjp is deprecated")
    f = xmap(lambda x, y: jnp.cos(lax.dot(x, jnp.sin(y),
                                          precision=lax.Precision.HIGHEST)),
             in_axes=(['i', ...], {}), out_axes=['i', ...])
    x = jnp.arange(12, dtype=jnp.float32).reshape((3, 4)) / 100
    y = jnp.arange(20, dtype=jnp.float32).reshape((4, 5)) / 100
    jtu.check_grads(f, (x, y), order=2, modes=['fwd'])
    jtu.check_grads(f, (x, y), order=1, modes=['rev'])
    with self.assertRaises(AssertionError):
      # Second order reverse-mode differentiations seems to be broken,
      # likely due to the transpose of psum being defined incorrectly.
      jtu.check_grads(f, (x, y), order=2, modes=['rev'])

  def testAutodiffNoBroadcast(self):
    f = xmap(lambda x, y: jnp.cos(lax.dot(x, jnp.sin(y),
                                          precision=lax.Precision.HIGHEST)),
             in_axes=(['i', ...], [None, 'i']), out_axes=['i'])
    x = jnp.arange(12, dtype=jnp.float32).reshape((3, 4)) / 100
    y = jnp.arange(12, dtype=jnp.float32).reshape((4, 3)) / 100
    jtu.check_grads(f, (x, y), order=2)

  @jtu.with_and_without_mesh
  def testNamedShape(self, mesh, axis_resources):
    x = np.arange(4,)
    y = 2
    f = xmap(lambda x, y: (x + y, y * lax.axis_index('i')),
             in_axes=(['i', ...], {}),
             out_axes=(['i', ...], ['i', ...]),
             axis_resources=dict(axis_resources))
    z, w = f(x, y)
    self.assertEqual(z.aval.named_shape, {})
    self.assertEqual(w.aval.named_shape, {})

  @jtu.with_and_without_mesh
  def testBroadcast(self, mesh, axis_resources):
    x = jnp.asarray(2.0)
    f = xmap(lambda x: x, in_axes={}, out_axes=['i'],
             axis_sizes={'i': 4}, axis_resources=dict(axis_resources))
    self.assertAllClose(f(x), jnp.asarray([2.0, 2.0, 2.0, 2.0]))

  def testNestedBroadcast(self):
    x = jnp.asarray(2.0)
    f = xmap(lambda x: x, in_axes={}, out_axes=['i'], axis_sizes={'i': 4})
    g = xmap(f, in_axes={}, out_axes=['j', ...], axis_sizes={'j': 7})
    self.assertAllClose(g(x), jnp.tile(x.reshape((1, 1)), (7, 4)))

  @serial_loop('l', 4)
  def testLoopBasic(self):
    x = jnp.arange(16)
    y = xmap(lambda x: x + 4, in_axes=['i'], out_axes=['i'],
              axis_resources={'i': 'l'})(x)
    self.assertAllClose(y, x + 4)

  @jtu.with_mesh([('x', 2)])
  @serial_loop('l', 4)
  def testLoopWithMesh(self):
    x = jnp.arange(16)
    y = xmap(lambda x: x + 4, in_axes=['i'], out_axes=['i'],
              axis_resources={'i': ('x', 'l')})(x)
    self.assertAllClose(y, x + 4)

  def testLoopAnonBasic(self):
    x = jnp.arange(16)
    y = xmap(lambda x: x + 4, in_axes=['i'], out_axes=['i'],
              axis_resources={'i': SerialLoop(4)})(x)
    self.assertAllClose(y, x + 4)

  @jtu.with_mesh([('x', 2)])
  def testLoopAnonWithMesh(self):
    x = jnp.arange(16)
    y = xmap(lambda x: x + 4, in_axes=['i'], out_axes=['i'],
              axis_resources={'i': ('x', SerialLoop(4))})(x)
    self.assertAllClose(y, x + 4)

  def testLowerWithAbstractArgs(self):
    x = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    # Make sure this doesn't crash
    xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...]).lower(x)

  def testLowerCompile(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f_exe = f.lower(x).compile()
    self.assertAllClose(f_exe(x), f(x))

  def testLowerCompileInTreeMismatch(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f_exe = f.lower(x).compile()
    self.assertRaisesRegex(
        TypeError,
        'Function compiled with input pytree does not match the input pytree it'
        ' was called with',
        lambda: f_exe([x]))

  def testLowerCompileArgTypeMismatch(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    x_f32 = x.astype(jnp.float32)
    x_i32 = x.astype(jnp.int32)
    f_exe = f.lower(x_f32).compile()
    self.assertRaisesRegex(
        TypeError,
        r"Argument types differ .*"
        r"The mismatches are:\n"
        r"Argument 1/1 compiled with.*float32.*and called with.*int32.*",
      lambda: f_exe(x_i32))

  def testLowerAsText(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f = f.lower(x)
    self.assertIsInstance(f.as_text(), str)
    self.assertIsInstance(f.as_text(dialect='hlo'), str)
    self.assertIsInstance(f.as_text(dialect='stablehlo'), str)

  def testLowerCompilerIR(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f = f.lower(x)
    self.assertIsNotNone(f.compiler_ir())
    self.assertIsNotNone(f.compiler_ir(dialect='hlo'))
    self.assertIsNotNone(f.compiler_ir(dialect='stablehlo'))

  @jtu.with_mesh([('x', 2)])
  def testLowerPartitionsAttribute(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...],
             axis_resources={'i': 'x'})
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    hlo = f.lower(x).as_text(dialect='stablehlo')
    if maps.SPMD_LOWERING.value:
      self.assertIn("mhlo.num_partitions = 2", hlo)
      self.assertIn("mhlo.num_replicas = 1", hlo)
    else:
      self.assertIn("mhlo.num_partitions = 1", hlo)
      self.assertIn("mhlo.num_replicas = 2", hlo)

  def testLowerCompileCompilerIR(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f = f.lower(x).compile()
    self.assertIsNotNone(f.runtime_executable())

  def testLowerCompileAsText(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f = f.lower(x).compile()
    self.assertIsInstance(f.as_text(), (str, type(None)))

  def testLowerCostAnalysis(self):
    # TODO(b/261771737): add support for uncompiled cost analysis in C API.
    if "PJRT C API" in xla_bridge.get_backend().platform_version:
      raise SkipTest("C API does not support uncompiled cost analysis")
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f = f.lower(x)
    f.cost_analysis()  # doesn't raise

  def testLowerCompileCostAnalysis(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f = f.lower(x).compile()
    f.cost_analysis()  # doesn't raise

  def testLowerCompileMemoryAnalysis(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f = f.lower(x).compile()
    f.memory_analysis()  # doesn't raise

  def testLowerCompileExecutable(self):
    f = xmap(lambda x: x + 4, in_axes=['i', ...], out_axes=['i', ...])
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    f = f.lower(x).compile()
    self.assertIsNotNone(f.runtime_executable())

  def testNewCheckpoint(self):
    f = checkpoint(xmap(lambda x: x, in_axes=['i', ...], out_axes=['i', ...]))
    self.assertAllClose(jax.grad(lambda x: f(x).sum())(jnp.arange(3.)), jnp.ones(3))

  def testNewCheckpointNonlinearWithPolicy(self):
    raise SkipTest("fails!")  # TODO(mattjj,apaszke): residual outvars problem
    f = checkpoint(xmap(lambda x: jnp.sin(jnp.sin(x)), in_axes=['i', ...],
                        out_axes=['i', ...]),
                   policy=lambda prim, *_, **__: str(prim) == 'sin')
    jax.grad(lambda x: f(x).sum())(jnp.arange(3.))  # TODO crashes!


@jtu.pytest_mark_if_available('multiaccelerator')
class XMapTestSPMD(SPMDTestMixin, XMapTest):
  """Re-executes all basic tests with the SPMD partitioner enabled"""

  skipped_tests = {
    "CollectivePermute2D"  # vmap of multidimensional permute not implemented yet
  }

  def setUp(self):
    for skipped_name in self.skipped_tests:
      if skipped_name in self._testMethodName:
        raise SkipTest
    super().setUp()

  @jtu.with_mesh([('x', 2), ('y', 2), ('z', 2)])
  @jax.numpy_dtype_promotion('standard')
  def testNestedMeshSPMD(self):
    h = xmap(lambda y: (jnp.sin(y) * np.arange(y.size, dtype=float),
                        lax.psum(y, ('a', 'b', 'c'))),
             in_axes={0: 'c'}, out_axes=({1: 'c'}, {}),
             axis_resources={'c': 'z'})
    f = xmap(lambda x: h(x * 2),
             in_axes=[None, 'a', 'b', ...], out_axes=(['a', 'b', ...], {}),
             axis_resources={'a': 'x', 'b': 'y'})
    xshape = (8, 2, 4, 5)
    x = jnp.arange(math.prod(xshape), dtype=float).reshape(xshape)
    hlo = f.lower(x).compiler_ir(dialect="hlo").as_hlo_text()
    match = re.search(
        r'sharding={devices=\[([0-9,]+)\]<=\[[0-9,]+\](T\([0-9,]+\))?}', hlo
    )
    self.assertIsNot(match, None)
    tile_factors = [int(s) for s in match.group(1).split(',')]
    self.assertEqual(set(tile_factors), {1, 2})

  @jtu.with_mesh([('x', 2)])
  def testFixedSharding(self):
    # TODO(apaszke): Add support for extracting XLA computations generated by
    # xmap and make this less of a smoke test.
    try:
      jax.config.update("experimental_xmap_ensure_fixed_sharding", True)
      f = xmap(lambda x: jnp.sin(2 * jnp.sum(jnp.cos(x) + 4, 'i')),
               in_axes=['i'], out_axes={}, axis_resources={'i': 'x'})
      x = jnp.arange(20, dtype=jnp.float32)
      f(x)
    finally:
      jax.config.update("experimental_xmap_ensure_fixed_sharding", False)

  @jtu.with_mesh([('x', 2)])
  def testConstantsInLowering(self):
    h = xmap(partial(constant_introducing_p.bind, axis_name='i'),
             in_axes=['i'], out_axes=['i'], axis_resources={'i': 'x'})
    f = pjit(h, in_shardings=None, out_shardings=None)
    yp = 1 + jnp.arange(10, dtype=np.float32)
    self.assertAllClose(
      f(jnp.ones(20, dtype=np.float32)),
      jnp.concatenate([yp, yp]))


@jtu.pytest_mark_if_available('multiaccelerator')
class XMapTestManualSPMD(ManualSPMDTestMixin, XMapTestCase):
  @jtu.with_mesh([('x', 2)])
  def testBasic(self):
    f = lambda x: jnp.sin(jnp.cos(x) + x) * x
    fx = xmap(f, in_axes=['i'], out_axes=['i'], axis_resources={'i': 'x'})
    x = jnp.arange(20, dtype=jnp.float32)
    self.assertAllClose(fx(x), f(x))

  @jtu.with_mesh([('x', 2)])
  def testReplicated(self):
    # TODO(apaszke): This seems to be failing if I try to have a replicated and a mapped argument?
    f = lambda x: jnp.sin(jnp.cos(x) + x) * x
    fx = xmap(f, in_axes=[...], out_axes=[...], axis_sizes={'i': 4}, axis_resources={'i': 'x'})
    x = jnp.arange(20, dtype=jnp.float32)
    self.assertAllClose(fx(x), f(x))

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testInPJit(self):
    f = xmap(lambda x: jnp.sin(x) + x, in_axes=['i'], out_axes=['i'], axis_resources={'i': 'x'})
    h = pjit(lambda x: f(x * x) + x, in_shardings=P('y'), out_shardings=None)
    x = jnp.arange(20, dtype=jnp.float32)
    self.assertAllClose(h(x), jnp.sin(x * x) + x * x + x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testInPJitReplicated(self):
    f = xmap(lambda x: jnp.sin(x) + x, in_axes={}, out_axes={}, axis_sizes={'i': 4}, axis_resources={'i': 'x'})
    h = pjit(lambda x: f(x * x) + x, in_shardings=P('y'), out_shardings=None)
    x = jnp.arange(20, dtype=jnp.float32)
    self.assertAllClose(h(x), jnp.sin(x * x) + x * x + x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testNestedConstraint(self):
    # TODO(b/219691408): Using P('y') instead of P() causes an XLA crash!
    fimpl = lambda x: jax.lax.with_sharding_constraint(jnp.sin(x), P()) + x
    f = xmap(fimpl, in_axes=['i', ...], out_axes=['i', ...], axis_resources={'i': 'x'})
    h = pjit(lambda x: f(x * x) + x, in_shardings=P('y'), out_shardings=None)
    x = jnp.arange(20, dtype=jnp.float32).reshape(4, 5)
    self.assertAllClose(h(x), jnp.sin(x * x) + x * x + x)

  @parameterized.named_parameters(
    {'testcase_name': name, 'mesh': mesh}
    for name, mesh in (
      ('1d', (('x', 2),)),
      ('2d', (('x', 2), ('y', 2))),
    ))
  @jtu.with_mesh_from_kwargs
  def testCollective(self, mesh):
    all_axes = tuple(axis[0] for axis in mesh)
    f = xmap(lambda x: lax.psum(x, 'i'), in_axes=['i', 'j'], out_axes=['j'],
             axis_resources=dict(zip('ij', all_axes)))
    h = pjit(lambda x: f(x * x), in_shardings=P(*all_axes), out_shardings=None)
    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    self.assertAllClose(h(x), (x * x).sum(0))

  @parameterized.named_parameters(
  {'testcase_name': name, 'mesh': mesh}
  for name, mesh in (
          ('1d', (('x', 2),)),
  ))
  @jtu.with_mesh_from_kwargs
  def testAllGather(self, mesh):
    # try hard_xmap variant, mapping across leading axes
    x = jnp.arange(8).reshape(2, 4)
    f = xmap(lambda x: lax.all_gather(x, 'i', axis=0, tiled=True),
             in_axes=['i', None], out_axes=[None],
             axis_resources={'i': 'x'})
    h = pjit(f, in_shardings=P('x', None), out_shardings=P(None))(x)
    self.assertArraysEqual(h.addressable_data(0), x.reshape(8))

  @parameterized.named_parameters(
  {'testcase_name': name, 'mesh': mesh}
  for name, mesh in (
          ('1d', (('x', 2),)),
  ))
  @jtu.with_mesh_from_kwargs
  def testReduceScatter(self, mesh):
    # try hard_xmap variant, mapping across leading axes
    x = jnp.arange(8).reshape(2, 4)
    f = xmap(lambda x: lax.psum_scatter(x, 'i', scatter_dimension=0, tiled=True),
             in_axes=[None, None], out_axes=['i', None, None], axis_sizes={'i': 2},
             axis_resources={'i': 'x'})
    h = pjit(
        lambda x: f(x).reshape((2, 4)),
        in_shardings=P(None, None),
        out_shardings=P('x', None),
    )(x)

    self.assertArraysEqual(h.addressable_data(0).reshape(4), x[0, :] * 2)

  @jtu.with_mesh([('x', 2)])
  def testBareXmapCollective(self):
    x = jnp.arange(20, dtype=jnp.float32).reshape(4, 5)

    y = xmap(lambda x: lax.psum(x, 'i'),
             in_axes=['i', ...], out_axes=[...], axis_resources={'i': 'x'})(x)
    self.assertAllClose(x.sum(0), y)

  @jtu.with_mesh([('x', 2)])
  def testPPermute(self):
    n = 2
    x = jnp.arange(n * 5, dtype=jnp.float32).reshape(n, 5)

    f = xmap(lambda x: lax.ppermute(x, 'i', perm=[(j, (j + 1) % n) for j in range(n)]),
             in_axes=['i', ...], out_axes=['i', ...], axis_resources={'i': 'x'})
    g = pjit(f, in_shardings=P('x'), out_shardings=P('x'))
    self.assertAllClose(g(x), x[::-1])

  @jtu.with_mesh([('x', 2)])
  def testConstantsInLowering(self):
    h = xmap(partial(constant_introducing_p.bind, axis_name='i'),
             in_axes=['i'], out_axes=['i'], axis_resources={'i': 'x'})
    f = pjit(h, in_shardings=None, out_shardings=None)
    yp = 1 + jnp.arange(10, dtype=np.float32)
    self.assertAllClose(
      f(jnp.ones(20, dtype=np.float32)),
      jnp.concatenate([yp, yp]))


@jtu.pytest_mark_if_available('multiaccelerator')
class NamedNumPyTest(XMapTestCase):

  @jtu.sample_product(
    reduction=(jnp.sum, jnp.max, jnp.min, jnp.mean, jnp.var, jnp.std,
               jscipy.special.logsumexp),
    axes=(0, 'i', (1,), ('i',), (0, 1), (0, 'i'), ('i', 0)),
    mapped_axis=range(3),
  )
  def testReductions(self, reduction, axes, mapped_axis):
    axes_t = axes if isinstance(axes, tuple) else (axes,)
    ref_red = partial(reduction,
                      axis=tuple(mapped_axis if a == 'i' else a + (a >= mapped_axis)
                                 for a in axes_t))
    mapped_axis_after_red = mapped_axis - sum(axis < mapped_axis if axis != 'i' else 0
                                              for axis in axes_t)
    xmap_red = xmap(lambda x: reduction(x, axes),
                    in_axes={mapped_axis: 'i'},
                    out_axes=({} if 'i' in axes_t else {mapped_axis_after_red: 'i'}))

    rng = self.rng()
    x = rng.randn(2, 5, 6)
    self.assertAllClose(ref_red(x), xmap_red(x))


@jtu.pytest_mark_if_available('multiaccelerator')
class NamedRandomTest(XMapTestCase):

  SAMPLERS = [
    ("Uniform", jax.random.uniform),
    ("Normal", jax.random.normal),
    ("Bernoulli", partial(jax.random.bernoulli, p=0.5)),
    ("TruncatedNormal", partial(jax.random.truncated_normal, lower=-2, upper=2)),
  ]

  @parameterized.parameters(*SAMPLERS)
  def testSamplerSharding(self, distr_name, distr_sample):
    def sample(shape, map_size):
      return xmap(lambda: distr_sample(jax.random.PRNGKey(0), shape=shape),
                  in_axes=(), out_axes=[None, 'i', ...], axis_sizes={'i': map_size})()
    replicated = sample((3,), 4)
    self.assertTrue((replicated[:,[0]] == replicated).all())
    sharded = sample(NamedShape(3, i=4), 4)
    self.assertFalse((sharded[:,[0]] == sharded[:,1:]).all(1).any())
    error = "The shape of axis i was specified as 4, but it really is 5"
    with self.assertRaisesRegex(ValueError, error):
      sample(NamedShape(3, i=4), 5)

  @jtu.sample_product(
    [dict(distr_name=name, distr_sample=sample)
     for name, sample in SAMPLERS],
    [dict(axis_resources=tuple(axis_resources.items()), mesh=tuple(mesh))
     for axis_resources, mesh in schedules({'i': 4, 'j': 6})],
  )
  @jtu.with_mesh_from_kwargs
  def testSamplerResourceIndependence(self, distr_name, distr_sample,
                                      axis_resources, mesh):
    def sample(axis_resources):
      return xmap(lambda: distr_sample(jax.random.PRNGKey(0), shape=NamedShape(3, i=4, j=6)),
                  in_axes=(), out_axes=['i', 'j', ...], axis_sizes={'i': 4, 'j': 6},
                  axis_resources=axis_resources)()
    self.assertAllClose(sample({}), sample(dict(axis_resources)))


@jtu.pytest_mark_if_available('multiaccelerator')
class NamedNNTest(XMapTestCase):

  def testOneHot(self):
    f = xmap(lambda x: jax.nn.one_hot(jnp.array([1, 2, 0], dtype='int32'), 3, axis='i'),
             in_axes=['i', ...], out_axes=['i', ...])
    expected = jnp.array([[0., 1., 0.],
                          [0., 0., 1.],
                          [1., 0., 0.]]).T
    self.assertAllClose(f(jnp.ones(3, dtype='int32')), expected, check_dtypes=False)

  def testOneHotOutOfBound(self):
    f = xmap(lambda x: jax.nn.one_hot(jnp.array([-1, 3], dtype='int32'), 3, axis='i'),
             in_axes=['i', ...], out_axes=['i', ...])
    self.assertAllClose(f(jnp.ones(3, dtype='int32')), jnp.zeros((3, 2)))

  def testOneHotAxisSizeMismatch(self):
    f = xmap(lambda x: jax.nn.one_hot(jnp.array([-1, 3], dtype='int32'), 3, axis='i'),
             in_axes=['i', ...], out_axes=['i', ...])
    with self.assertRaisesRegex(ValueError, "to match the size of axis i, but 3 != 5"):
      f(jnp.ones(5, dtype='int32'))

  @jtu.sample_product(
    [dict(map_in=map_in, map_out=map_out)
     for map_in, map_out in [(True, False), (False, True), (True, True)]],
    fan=['fan_in', 'fan_out', 'fan_avg'],
    distr=['uniform', 'normal', 'truncated_normal'],
  )
  def testVarianceScaling(self, map_in, map_out, fan, distr):
    shape = (80, 50, 7)
    fan_in, fan_out = nn_initializers._compute_fans(NamedShape(*shape), 0, 1)
    key = jax.random.PRNGKey(1)
    base_scaling = partial(jax.nn.initializers.variance_scaling, 100, fan, distr)
    ref_sampler = lambda: base_scaling(in_axis=0, out_axis=1)(key, shape)
    if map_in and map_out:
      out_axes = ['i', 'o', ...]
      named_shape = NamedShape(shape[2], i=shape[0], o=shape[1])
      xmap_sampler = lambda: base_scaling(in_axis='i', out_axis='o')(key, named_shape)
    elif map_in:
      out_axes = ['i', ...]
      named_shape = NamedShape(shape[1], shape[2], i=shape[0])
      xmap_sampler = lambda: base_scaling(in_axis='i', out_axis=0)(key, named_shape)
    elif map_out:
      out_axes = [None, 'o', ...]
      named_shape = NamedShape(shape[0], shape[2], o=shape[1])
      xmap_sampler = lambda: base_scaling(in_axis=0, out_axis='o')(key, named_shape)
    mapped_sampler = xmap(xmap_sampler,
                          in_axes=(), out_axes=out_axes,
                          axis_sizes={'i': shape[0], 'o': shape[1]})
    self.assertAllClose(jnp.var(mapped_sampler()), jnp.var(ref_sampler()),
                        atol=1e-4, rtol=2e-2)


@jtu.pytest_mark_if_available('multiaccelerator')
class XMapArrayTest(XMapTestCase):

  def test_basic(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P('x', 'y')
    input_array, input_data = create_array(global_input_shape, global_mesh,
                                           mesh_axes)

    with global_mesh:
      f = maps.xmap(
            lambda x: x,
            in_axes=({0: "a", 1: "b"}),
            out_axes=({0: "a", 1: "b"}),
            axis_resources={"a": "x", "b": "y"})

      out = f(input_array)
      self.assertIsInstance(out, array.ArrayImpl)
      self.assertEqual(out.shape, (8, 2))
      self.assertEqual(out.addressable_shards[0].data.shape, (2, 1))
      self.assertDictEqual(out.sharding.mesh.shape, {'x': 4, 'y': 2})
      for s in out.addressable_shards:
        self.assertArraysEqual(s.data, input_data[s.index])

  def test_xmap_array_double_input(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    a1, input_data = create_array(global_input_shape, global_mesh, P('x'))
    a2, _ = create_array(global_input_shape, global_mesh, P('y'))

    with global_mesh:
      f = maps.xmap(
            lambda x, y: (x @ x.T, y @ y.T),
            in_axes=({0: "a"}, ["c", ...]),
            out_axes=({0: "a"}, ["c", ...]),
            axis_resources={"a": "x", "c": "y"})

      expected_matrix_mul = np.diagonal(input_data @ input_data.T)
      out1, out2 = f(a1, a2)

      self.assertIsInstance(out1, array.ArrayImpl)
      self.assertEqual(out1.shape, (8,))
      self.assertEqual(out1.addressable_shards[0].data.shape, (2,))
      self.assertDictEqual(out1.sharding.mesh.shape, {'x': 4, 'y': 2})
      for s in out1.addressable_shards:
        self.assertArraysEqual(s.data, expected_matrix_mul[s.index])

      self.assertIsInstance(out2, array.ArrayImpl)
      self.assertEqual(out2.shape, (8,))
      self.assertEqual(out2.addressable_shards[0].data.shape, (4,))
      self.assertDictEqual(out2.sharding.mesh.shape, {'x': 4, 'y': 2})
      for s in out2.addressable_shards:
        self.assertArraysEqual(s.data, expected_matrix_mul[s.index])

  def test_xmap_array_sharding_mismatch(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P('x', 'y')
    input_array, _ = create_array(global_input_shape, global_mesh, mesh_axes)

    with global_mesh:
      f = maps.xmap(
            lambda x: x @ x.T,
            in_axes=({0: "a"}),
            out_axes=({0: "a"}),
            axis_resources={"a": "x"})
      with self.assertRaisesRegex(
          ValueError,
          ('Got an input Array to xmap with different partitioning than '
            'specified in xmap. The partitioning must match.')):
        f(input_array)

  def test_can_stage_and_interpret_xmap_with_constants(self):
    def f(x):
      # Create a jaxpr with a constant
      return x + np.ones(x.shape, x.dtype)

    def outer(x):
      return xmap(f, in_axes=({0: 'batch'},), out_axes={0: 'batch'})(x)

    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    jaxpr = jax.make_jaxpr(outer)(x)
    [out] = jax.core.jaxpr_as_fun(jaxpr)(x)
    self.assertAllClose(out, x + 1)


@jtu.pytest_mark_if_available('multiaccelerator')
class NewPrimitiveTest(XMapTestCase):

  def testGatherPositional(self):
    x = jnp.arange(27).reshape((9, 3))
    idx = jnp.array([1, 2, 1, 0]).reshape((2, 2))
    self.assertAllClose(pgather(x, idx, 0), x[idx.ravel()].reshape((2, 2, 3)))

    x_explode = x.reshape((3, 3, 3))
    self.assertAllClose(pgather(x, idx, 0), pgather(x_explode, idx, (0, 1)))

  @jtu.with_and_without_mesh
  def testGather(self, mesh, axis_resources):
    if axis_resources and not maps.SPMD_LOWERING.value:
      raise SkipTest("pgather over mesh axes without SPMD lowering not implemented")
    x = jnp.arange(12, dtype=np.float32).reshape((4, 3))
    y = jnp.arange(35).reshape((5, 7)) % 3
    f = xmap(lambda src, idx: pgather(src, idx, 'j'),
             in_axes=(['i', 'j'], ['k', 'm']),
             out_axes=['i', 'k', 'm'],
             axis_resources=dict(axis_resources))
    f_ref = lambda x, y: x[:, y.reshape((-1,))].reshape((4, 5, 7))
    self.assertAllClose(f(x, y), f_ref(x, y))


@jtu.pytest_mark_if_available('multiaccelerator')
class NewPrimitiveTestSPMD(SPMDTestMixin, NewPrimitiveTest):
  pass


AxisIndices = tuple[int, ...]
MatchedAxisIndices = tuple[AxisIndices, AxisIndices]
AxisNames = tuple[str, ...]

class PdotTestSpec:
  # The axis indices stored by a PdotTestSpec are all positional indices
  # *before* taking mapping into account.
  map_cont: MatchedAxisIndices
  pos_cont: MatchedAxisIndices
  map_batch: MatchedAxisIndices
  pos_batch: MatchedAxisIndices
  all_names: AxisNames
  contract_names: AxisNames
  batch_names: AxisNames

  def __init__(self, map_cont, pos_cont, map_batch, pos_batch):
    self.map_cont = map_cont
    self.pos_cont = pos_cont
    self.map_batch = map_batch
    self.pos_batch = pos_batch

    names = gen_axis_names()
    self.contract_names = [next(names) for _ in range(len(map_cont[0]))]
    self.batch_names = [next(names) for _ in range(len(map_batch[0]))]
    self.all_names = self.contract_names + self.batch_names

  @property
  def dot_general_dim_nums(self):
    lhs_contract = (*self.map_cont[0], *self.pos_cont[0])
    rhs_contract = (*self.map_cont[1], *self.pos_cont[1])
    lhs_batch = (*self.map_batch[0], *self.pos_batch[0])
    rhs_batch = (*self.map_batch[1], *self.pos_batch[1])
    return (lhs_contract, rhs_contract), (lhs_batch, rhs_batch)

  @property
  def pos_contract_after_mapping(self):
    lhs = [i - sum(j < i for j in self._lhs_mapped) for i in self.pos_cont[0]]
    rhs = [i - sum(j < i for j in self._rhs_mapped) for i in self.pos_cont[1]]
    return (lhs, rhs)

  @property
  def pos_batch_after_mapping(self):
    lhs = [i - sum(j < i for j in self._lhs_mapped) for i in self.pos_batch[0]]
    rhs = [i - sum(j < i for j in self._rhs_mapped) for i in self.pos_batch[1]]
    return (lhs, rhs)

  @property
  def _lhs_mapped(self):
    return {*self.map_cont[0], *self.map_batch[0]}

  @property
  def _rhs_mapped(self):
    return {*self.map_cont[1], *self.map_batch[1]}

  @property
  def lhs_in_axes(self):
    axis_indices = [*self.map_cont[0], *self.map_batch[0]]
    return dict(zip(axis_indices, self.all_names))

  @property
  def rhs_in_axes(self):
    axis_indices = [*self.map_cont[1], *self.map_batch[1]]
    return dict(zip(axis_indices, self.all_names))

def all_pdot_specs(lhs_shape, rhs_shape):
  for matching in axis_matchings(lhs_shape, rhs_shape):
    for lists in partitions(matching, 4):
      yield PdotTestSpec(*map(unzip2, lists))

def axis_matchings(lhs_shape, rhs_shape):
  def helper(start, exc1, exc2):
    yield ()
    for i in range(start, len(lhs_shape)):
      d1 = lhs_shape[i]
      if i not in exc1:
        for j, d2 in enumerate(rhs_shape):
          if d1 == d2 and j not in exc2:
            for matches in helper(i + 1, exc1 | {i}, exc2 | {j}):
              yield ((i, j), *matches)
  return helper(0, set(), set())

def gen_axis_names():
  names = 'ijkl'
  for n in it.count(1):
    for chars in product(names, repeat=n):
      yield ''.join(chars)


def schedules_from_pdot_spec(
    spec: PdotTestSpec, lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]
    ) -> Generator[tuple[AxisResources, jtu.MeshSpec], None, None]:
  logical_sizes = {
      name: shape[ax]
      for shape, in_axes in [(lhs_shape, spec.lhs_in_axes),
                             (rhs_shape, spec.rhs_in_axes)]
      for ax, name in in_axes.items()}
  yield from schedules(logical_sizes)


@jtu.pytest_mark_if_available('multiaccelerator')
class PDotTests(XMapTestCase):

  @jtu.with_mesh([('r1', 2)])
  def testPdotBasic(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    f_mapped = xmap(f,
                    in_axes=({1: 'i'}, {0: 'i'}),
                    out_axes={},
                    axis_resources={'i': 'r1'})

    rng = self.rng()
    x = rng.randn(3, 8)
    y = rng.randn(8, 5)

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.dot(x, y))

  @jtu.with_mesh([('r1', 2)])
  @jax.default_matmul_precision("float32")
  def testPdotBatching(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = self.rng()
    x = rng.randn(2, 3, 8)
    y = rng.randn(2, 8, 5)

    f_mapped = xmap(f,
                    in_axes=({0: 'j', 2: 'i'}, {0: 'j', 1: 'i'}),
                    out_axes=['j', ...],
                    axis_resources={'i': 'r1'})

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.einsum('nij,njk->nik', x, y))

  @jtu.with_mesh([('r1', 2)])
  @jax.default_matmul_precision("float32")
  def testPdotBatchingShardUncontractedDim(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = self.rng()
    x = rng.randn(2, 3, 8)
    y = rng.randn(2, 8, 5)

    f_mapped = xmap(f,
                    in_axes=({0: 'j', 2: 'i'}, {0: 'j', 1: 'i'}),
                    out_axes=['j', ...],
                    axis_resources={'j': 'r1'})

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.einsum('nij,njk->nik', x, y))

  @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
       "testcase_name": f"_{next(test_counter)}",
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "pdot_spec": pdot_spec,
       "axis_resources": axis_resources, "mesh_data": mesh_data
    } for test_counter in [it.count()]
      for lhs_shape, rhs_shape in s(product([(2,), (2, 4, 2, 1)], repeat=2))
      for pdot_spec in s(all_pdot_specs(lhs_shape, rhs_shape))
      for axis_resources, mesh_data in s(schedules_from_pdot_spec(
          pdot_spec, lhs_shape, rhs_shape))
  )))
  @jax.default_matmul_precision("float32")
  def testPdotSystematic(self, lhs_shape, rhs_shape, pdot_spec, axis_resources,
                         mesh_data):
    rng = jtu.rand_default(self.rng())
    lhs = rng(lhs_shape, np.float32)
    rhs = rng(rhs_shape, np.float32)

    def pdot_fun(x, y):
      # print(f'pdot(x:{x.aval.str_short()}, y:{y.aval.str_short()},\n'
      #       f'     axis_name={contract_names},\n'
      #       f'     pos_contract={spec.pos_contract_after_mapping}\n'
      #       f'     pos_batch={spec.pos_batch_after_mapping})')
      return jax.lax.pdot(x, y, axis_name=pdot_spec.contract_names,
                          pos_batch=pdot_spec.pos_batch_after_mapping,
                          pos_contract=pdot_spec.pos_contract_after_mapping)

    fun = xmap(pdot_fun, in_axes=(pdot_spec.lhs_in_axes, pdot_spec.rhs_in_axes),
               out_axes=[*pdot_spec.batch_names, ...],
               axis_resources=axis_resources)

    with jtu.with_mesh(mesh_data):
      result = fun(lhs, rhs)

    expected = lax.dot_general(lhs, rhs, pdot_spec.dot_general_dim_nums)
    self.assertAllClose(result, expected, check_dtypes=False)

  @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
       "testcase_name": f"_{next(test_counter)}",
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "pdot_spec": pdot_spec,
       "axis_resources": axis_resources, "mesh_data": mesh_data
    } for test_counter in [it.count()]
      for lhs_shape, rhs_shape in s(product([(2,), (2, 4, 2, 1)], repeat=2))
      for pdot_spec in s(all_pdot_specs(lhs_shape, rhs_shape))
      for axis_resources, mesh_data in s(schedules_from_pdot_spec(
          pdot_spec, lhs_shape, rhs_shape))
  )))
  @jax.default_matmul_precision("float32")
  def testPdotVJPSystematic(self, lhs_shape, rhs_shape, pdot_spec,
                            axis_resources, mesh_data):
    rng = jtu.rand_default(self.rng())
    lhs = rng(lhs_shape, np.float32)
    rhs = rng(rhs_shape, np.float32)

    expected_out, ref_vjp = jax.vjp(
        lambda x, y: lax.dot_general(x, y, pdot_spec.dot_general_dim_nums),
        lhs, rhs)
    out_bar = rng(expected_out.shape, np.float32)
    expected_lhs, expected_rhs = ref_vjp(out_bar)

    def pdot_fun(x, y, out_bar):
      pdot = partial(jax.lax.pdot,
                     axis_name=pdot_spec.contract_names,
                     pos_batch=pdot_spec.pos_batch_after_mapping,
                     pos_contract=pdot_spec.pos_contract_after_mapping)
      _, pdot_vjp = jax.vjp(pdot, x, y)
      return pdot_vjp(out_bar)

    fun = xmap(pdot_fun,
               in_axes=(pdot_spec.lhs_in_axes, pdot_spec.rhs_in_axes,
                        [*pdot_spec.batch_names, ...]),
               out_axes=(pdot_spec.lhs_in_axes, pdot_spec.rhs_in_axes),
               axis_resources=axis_resources)

    with jtu.with_mesh(mesh_data):
      lhs_bar, rhs_bar = fun(lhs, rhs, out_bar)

    self.assertAllClose(lhs_bar, expected_lhs, check_dtypes=False)
    self.assertAllClose(rhs_bar, expected_rhs, check_dtypes=False)

  def test_xeinsum_vector_dot(self):
    rng = self.rng()
    x = rng.randn(3)
    y = rng.randn(3)
    out = xmap(partial(lax.xeinsum, '{i},{i}->'),
               in_axes=(['i'], ['i']), out_axes=[])(x, y)
    expected = np.einsum('i,i->', x, y)
    self.assertAllClose(out, expected, check_dtypes=False)

  def test_xeinsum_outer_product(self):
    rng = self.rng()
    x = rng.randn(3)
    y = rng.randn(3)
    out = xmap(partial(lax.xeinsum, '{i},{j}->{i,j}'),
               in_axes=(['i'], ['j']), out_axes=['i', 'j'])(x, y)
    expected = np.einsum('i,j->ij', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)

  @jax.default_matmul_precision("float32")
  def test_xeinsum_matmul(self):
    rng = self.rng()
    x = rng.randn(3, 4)
    y = rng.randn(4, 5)

    def check(spec):
      out = xmap(partial(lax.xeinsum, spec),
                 in_axes=(['i', 'j'], ['j', 'k']),
                 out_axes=['i', 'k'])(x, y)
      expected = np.einsum('ij,jk->ik', x, y)
      self.assertAllClose(out, expected, check_dtypes=True)
    check('{i,j},{j,k}->{i,k}')
    check('{i,j},{k,j}->{k,i}')  # order of named axes in the spec doesn't matter!
    check('{j},{k,j}->{k}')
    check('{i,j},{j}->{i}')
    check('{j},{j}->{}')

  def test_xeinsum_no_named_axes_vector_dot(self):
    rng = self.rng()
    x = rng.randn(3)
    y = rng.randn(3)
    out = jnp.einsum('i,i->', x, y)
    expected = np.einsum('i,i->', x, y)
    self.assertAllClose(out, expected, check_dtypes=False)

  def test_xeinsum_no_named_axes_batch_vector_dot(self):
    rng = self.rng()
    x = rng.randn(3, 2)
    y = rng.randn(3, 2)
    out = lax.xeinsum('ij,ij->i', x, y)
    expected = np.einsum('ij,ij->i', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)

  @jax.default_matmul_precision("float32")
  def test_xeinsum_no_named_axes_batch_matmul(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3, 5, 4)
    y = rng.randn(3, 4, 2)
    out = lax.xeinsum('bij,bjk->bik', x, y)
    expected = np.einsum('bij,bjk->bik', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)

  def test_xeinsum_no_named_axes_reduce_sum(self):
    rng = self.rng()
    x = rng.randn(3)
    y = rng.randn()
    out = lax.xeinsum('i,->', x, y)
    expected = np.einsum('i,->', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)


  @jax.default_matmul_precision("float32")
  def test_xeinsum_no_named_axes_reduce_and_contract(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3, 5, 4)
    y = rng.randn(2, 4, 2)
    out = lax.xeinsum('bij,cjk->ik', x, y)
    expected = np.einsum('bij,cjk->ik', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)

  @jax.default_matmul_precision("float32")
  def test_xeinsum_named_axes_reduce(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3, 4)
    y = rng.randn(5,)

    def check(spec):
      out = xmap(partial(lax.xeinsum, spec),
                 in_axes=(['i', 'j'], ['k']),
                 out_axes=['i', 'k'])(x, y)
      expected = np.einsum('ij,k->ik', x, y)
      self.assertAllClose(out, expected, check_dtypes=True)
    check('{i,j},{k}->{i,k}')

  @jtu.with_mesh([('x', 2), ('y', 2)])
  @jax.default_matmul_precision("float32")
  def test_xeinsum_named_axes_reduce_with_mesh(self):
    rng = np.random.RandomState(0)
    x = rng.randn(6, 4)
    y = rng.randn(8,)

    def check(spec):
      out = xmap(partial(lax.xeinsum, spec),
                 in_axes=(['i', 'j'], ['k']),
                 out_axes=['i', 'k'],
                 axis_resources={'i': 'x', 'k': 'y'})(x, y)
      expected = np.einsum('ij,k->ik', x, y)
      self.assertAllClose(out, expected, check_dtypes=True)

    check('{i,j},{k}->{i,k}')
    check('{i,j},{k}->{k,i}')  # order of named axes in the spec doesn't matter!
    check('{j,i},{k}->{i,k}')
    check('{j,i},{k}->{k,i}')

  @jtu.with_mesh([('x', 2), ('y', 2)])
  @jax.default_matmul_precision("float32")
  def test_xeinsum_named_axes_batch_matmul_with_mesh(self):
    rng = np.random.RandomState(0)
    x = rng.randn(8, 3, 4)
    y = rng.randn(8, 4, 5)

    def check(spec):
      out = xmap(partial(lax.xeinsum, spec),
                 in_axes=(['b', 'i', 'j'], ['b', 'j', 'k']),
                 out_axes=['b', 'i', 'k'],
                 axis_resources={'b': 'x', 'j': 'y'})(x, y)
      expected = np.einsum('bij,bjk->bik', x, y)
      self.assertAllClose(out, expected, check_dtypes=True)

    check('{b,i,j},{b,j,k}->{b,i,k}')
    check('{j,i,b},{j,b,k}->{i,b,k}')  # order of named axes in the spec doesn't matter!

  @jtu.with_mesh([('x', 2), ('y', 2)])
  @jax.default_matmul_precision("float32")
  def test_xeinsum_named_axes_unary_reduce_with_mesh(self):
    rng = np.random.RandomState(0)
    x = rng.randn(8, 6, 4)

    def check(spec):
      out = xmap(partial(lax.xeinsum, spec),
                 in_axes=['b', 'i', 'j'],
                 out_axes=['b'],
                 axis_resources={'b': 'x', 'i': 'y'})(x)
      expected = np.einsum('bij->b', x)
      self.assertAllClose(out, expected, check_dtypes=True)

    check('{b,i,j}->{b}')
    check('{b,j,i}->{b}')  # order of named axes in the spec doesn't matter!
    check('{i,j,b}->{b}')

  @jtu.with_mesh([('x', 2), ('y', 2)])
  @jax.default_matmul_precision("float32")
  def test_xeinsum_mixed_axes_unary_reduce_with_mesh(self):
    rng = np.random.RandomState(0)
    x = rng.randn(8, 6, 4, 5)

    def check(spec):
      out = xmap(partial(lax.xeinsum, spec),
                 in_axes=['b', 'i', ...],
                 out_axes=['b', ...],
                 axis_resources={'b': 'x', 'i': 'y'})(x)
      expected = np.einsum('bijk->bk', x)
      self.assertAllClose(out, expected, check_dtypes=True)

    check('jk{i,b}->k{b}')


@jtu.pytest_mark_if_available('multiaccelerator')
@jtu.with_config(jax_legacy_prng_key="allow")
class XMapErrorTest(jtu.JaxTestCase):

  @jtu.with_mesh([('x', 2)])
  def testRepeatedAxisResource(self):
    def f(v):
      return v * 4
    with self.assertRaisesRegex(ValueError, r"distinct resources.*specified \('x', 'x'\) for axis a"):
      xmap(f, in_axes=['a', ...], out_axes=['a', ...],
           axis_resources={'a': ('x', 'x')})

  @jtu.with_mesh([('y', 2)])
  def testUndefinedAxisResource(self):
    error = re.escape(
        r"In-scope resources are insufficient to execute the xmapped function. "
        r"The missing resources are: {'x'}")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x: x, in_axes=['a', ...], out_axes=['a', ...],
           axis_resources={'a': 'x'})(jnp.zeros((4,)))

  @jtu.with_mesh([('x', 2)])
  def testNestedDifferentResources(self):
    @partial(xmap, in_axes={0: 'a'}, out_axes={0: 'a'}, axis_resources={'a': 'x'})
    def f(x):
      with jax.sharding.Mesh(np.empty((), dtype=np.object_), ()):
        @partial(xmap, in_axes={0: 'b'}, out_axes={0: 'b'})
        def h(x):
          return x
        return h(x)
    xshape = (2, 5, 6)
    x = jnp.arange(math.prod(xshape)).reshape(xshape)
    with self.assertRaisesRegex(RuntimeError,
                                "Changing the physical mesh is not allowed.*"):
      f(x)

  def testEmptyArgumentTrees(self):
    with self.assertRaisesRegex(ValueError, "Failed to infer size of axes: i."):
      xmap(lambda x: x, in_axes=['i', ...], out_axes=['i', ...])({})

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testAxesNotDivisibleByResources(self):
    with self.assertRaisesRegex(ValueError, r"Size of axis i \(5\) is not divisible.*"
                                            r"\(\('x', 'y'\), 4 in total\)"):
      xmap(lambda x: x, in_axes=['i', ...], out_axes=['i', ...],
           axis_sizes={'i': 5}, axis_resources={'i': ('x', 'y')})({})

  def testInconsistentAxisSizes(self):
    x5 = jnp.arange(5)
    x6 = jnp.arange(6)
    error = (r"The size of axis i was previously inferred to be 5, but found an "
             r"argument of shape \(6,\) with in_axes specification \['i', ...\]. "
             r"Shape mismatch occurs in dimension 0: 6 != 5")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x, y: x, in_axes=(['i', ...], ['i', ...]), out_axes=['i', ...])(x5, x6)
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x: x, in_axes=['i', ...], out_axes=['i', ...], axis_sizes={'i': 5})(x6)

  def testInAxesRankError(self):
    error = (r"One of xmap arguments has an in_axes specification of \['i', 'j', ...\], "
             r"which implies that it has at least 2 dimensions, but the argument has rank 1")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x: x, in_axes=['i', 'j', ...], out_axes=['j', 'i', ...])(jnp.ones((5,)))

  def testOutAxesRankError(self):
    error = (r"One of xmap outputs has an out_axes specification of {1: 'i'}, "
             r"which requires the result of the xmapped function to have at least "
             r"1 positional dimensions, but it only has 0")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x: x, in_axes=['i', ...], out_axes={1: 'i'})(jnp.ones((5,)))

  def testNegativeAxes(self):
    with self.assertRaisesRegex(ValueError, "xmap doesn't support negative axes in in_axes"):
      xmap(lambda x: x, in_axes={-1: 'i'}, out_axes={0: 'i'})(jnp.ones((5,)))
    with self.assertRaisesRegex(ValueError, "xmap doesn't support negative axes in out_axes"):
      xmap(lambda x: x, in_axes={0: 'i'}, out_axes={-1: 'i'})(jnp.ones((5,)))

  def testDictOutAxes(self):
    # see issue #6410
    out = xmap(lambda x: x, in_axes=[...], out_axes={"a": [...]})({"a": 1})
    self.assertEqual(out, {"a": 1})

  def testListAxesRankAssertion(self):
    error = (r"xmap argument has an in_axes specification of \['i', None\], which "
             r"asserts that it should be of rank 2, but the argument has rank 1 "
             r"\(and shape \(5,\)\)")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x: x, in_axes=['i', None], out_axes=['i', None])(jnp.ones((5,)))
    error = (r"xmap output has an out_axes specification of \['i', None\], which "
             r"asserts that it should be of rank 2, but the output has rank 3 "
             r"\(and shape \(5, 2, 2\)\)")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x: x.reshape((2, 2)),
           in_axes=['i', None], out_axes=['i', None])(jnp.ones((5, 4)))

  def testReturnExtraMappedAxes(self):
    raise SkipTest("reduce_axes on vjp is deprecated")
    fm = xmap(lambda x, y: x + y,
              in_axes=(['a', ...], ['b', ...]), out_axes=['a', ...])
    x = np.arange(12).reshape((4, 3))
    y = np.arange(6).reshape((2, 3))
    error = (r"One of xmap results has an out_axes specification of \['a', ...\], but "
             r"is actually mapped along more axes defined by this xmap call: b")
    with self.assertRaisesRegex(TypeError, error):
      fm(x, y)

  def testUndefinedOutAxis(self):
    error = (r"All axis names appearing in out_axes must also appear in "
             r"in_axes or axis_sizes, but the following are missing: {'c'}")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x, y: x + y,
           in_axes=(['a', ...], ['b', ...]), out_axes=['c', ...])

  @jtu.with_mesh([('x', 2)])
  def testUndefinedAxisInAxisResources(self):
    error = (r"All axes that were assigned resources have to appear in in_axes "
             r"or axis_sizes, but the following are missing: {'b'}")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x, y: x + y,
           in_axes=(['a', ...], ['a', ...]), out_axes=['a', ...],
           axis_resources={'b': 'x'})

  @jtu.with_mesh([('x', 2)])
  def testResourceConflictArgs(self):
    fm = xmap(lambda x: lax.psum(x, ('a', 'b')),
              in_axes=['a', 'b'], out_axes=[],
              axis_resources={'a': 'x', 'b': 'x'})
    x = np.arange(16).reshape(4, 4)
    error = (r"Axes `a` and `b` are both mapped to the resource `x`, but they "
             r"coincide in the named_shape of an input to an xmapped function "
             r"<lambda>")
    with self.assertRaisesRegex(JAXTypeError, error):
      fm(x)

  @jtu.with_mesh([('x', 2)])
  def testResourceConflictInner(self):
    fm = xmap(lambda x, y: x + y,
              in_axes=(['a', ...], ['b', ...]), out_axes=['a', 'b', ...],
              axis_resources={'a': 'x', 'b': 'x'})
    x = np.arange(12).reshape(4, 3)
    y = np.arange(6).reshape(2, 3)
    error = (r"Axes `a` and `b` are both mapped to the resource `x`, but they "
             r"coincide in the named_shape.*primitive add created at")
    with self.assertRaisesRegex(JAXTypeError, error):
      fm(x, y)

  @jtu.with_mesh([('x', 2)])
  def testResourceConflictOut(self):
    fm = xmap(lambda x, y: x,
              in_axes=(['a', ...], ['b', ...]), out_axes=['a', 'b', ...],
              axis_resources={'a': 'x', 'b': 'x'})
    x = np.arange(12).reshape(4, 3)
    y = np.arange(6).reshape(2, 3)
    error = (r"One of xmapped function \(<lambda>\) outputs is broadcast along axis "
             r"`b` which is assigned to resources `x`, but the output is already "
             r"partitioned along `x`, because its named shape contains `a`")
    with self.assertRaisesRegex(JAXTypeError, error):
      fm(x, y)

  @jtu.with_mesh([('x', 2)])
  def testResourceConflictNestArgs(self):
    f = xmap(lambda x: x, in_axes=['i'], out_axes=['i'], axis_resources={'i': 'x'})
    h = xmap(f, in_axes=['j', ...], out_axes=['j', ...], axis_resources={'j': 'x'})
    x = np.arange(16).reshape((4, 4))
    error = (r"Axes `i` and `j` are both mapped to the resource `x`, but they "
             r"coincide in the named_shape of an input to an xmapped function "
             r"<lambda> \(xmap called at .*\)")
    with self.assertRaisesRegex(JAXTypeError, error):
      h(x)

  @jtu.with_mesh([('x', 2)])
  def testResourceConflictNestInner(self):
    f = xmap(lambda x: lax.axis_index('i') + x,
             in_axes=[], out_axes=['i'], axis_sizes={'i': 4}, axis_resources={'i': 'x'})
    h = xmap(f, in_axes=['j', ...], out_axes=['j', ...], axis_resources={'j': 'x'})
    x = np.arange(4, dtype='int32')
    error = (r"Axes `i` and `j` are both mapped to the resource `x`, but they "
             r"coincide in the named_shape of a value returned from a primitive "
             r"add created at .*")
    with self.assertRaisesRegex(JAXTypeError, error):
      h(x)

  @jtu.with_mesh([('x', 2)])
  def testResourceConflictNestOut(self):
    f = xmap(lambda x: x,
             in_axes=[], out_axes=['i'], axis_sizes={'i': 4}, axis_resources={'i': 'x'})
    h = xmap(f, in_axes=['j', ...], out_axes=['j', ...], axis_resources={'j': 'x'})
    x = np.arange(4)
    error = (r"One of xmapped function \(<lambda>\) outputs is broadcast along "
             r"axis `i` which is assigned to resources `x`, but the output is "
             r"already partitioned along `x`, because its named shape contains `j`")
    with self.assertRaisesRegex(JAXTypeError, error):
      h(x)

  @serial_loop('l', 2)
  def testResourceConflictArgsLoop(self):
    fm = xmap(lambda x: x,
              in_axes=['a', 'b'], out_axes=['a', 'b'],
              axis_resources={'a': 'l', 'b': 'l'})
    x = np.arange(16).reshape(4, 4)
    error = (r"Axes `a` and `b` are both mapped to the resource `l`, but they "
             r"coincide in the named_shape of an input to an xmapped function "
             r"<lambda>")
    with self.assertRaisesRegex(JAXTypeError, error):
      fm(x)

  @serial_loop('l', 2)
  def testLoopCollectives(self):
    fm = xmap(lambda x: lax.psum(x, 'i'),
              in_axes=['i'], out_axes=[],
              axis_resources={'i': 'l'})
    x = np.arange(16)
    error = (r"Named axes with loop resources assigned to them cannot be "
             r"referenced inside the xmapped computation \(e.g. in "
             r"collectives\), but `i` violates that rule")
    with self.assertRaisesRegex(RuntimeError, error):
      fm(x)

  def testAxesMismatch(self):
    x = jnp.ones((4,))
    p = [['x'], ['x'], ['x']]
    xmap(lambda x: x, (p,), p)([x, x, x])  # OK
    xmap(lambda x: x, [p], p)([x, x, x])  # OK
    error = re.escape(
        r"xmap in_axes specification must be a tree prefix of the "
        r"corresponding value, got specification (['x'], ['x'], ['x']) for value "
        r"tree PyTreeDef((*, *)). Note that xmap in_axes that are "
        r"non-trivial pytrees should always be wrapped in a tuple representing "
        r"the argument list.")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x, y: x, p, p)(x, x)  # Error, but make sure we hint at tupling
    # TODO(apaszke): Disable implicit list casts and enable this
    # error = re.escape(
    # r"xmap in_axes specification must be a tree prefix of the "
    # r"corresponding value, got specification (['x'], ['x'], ['x']) for value "
    # r"tree PyTreeDef(([*, *, *],)). Note that xmap in_axes that "
    # r"are non-trivial pytrees should always be wrapped in a tuple representing "
    # r"the argument list. In particular, you're passing in a single argument "
    # r"which means that xmap in_axes might need to be wrapped in a "
    # r"singleton tuple.")
    # with self.assertRaisesRegex(ValueError, error):
    # xmap(lambda x: x, p, p)([x, x, x])  # Error, but make sure we hint at singleton tuple
    error = re.escape(
        r"xmap out_axes specification must be a tree prefix of the "
        r"corresponding value, got specification ([['x'], ['x'], ['x']], ['x']) for "
        r"value tree PyTreeDef([*, *, *]).")
    with self.assertRaisesRegex(ValueError, error):
      xmap(lambda x: x, (p,), (p, ['x']))([x, x, x])  # Error, we raise a generic tree mismatch message


@jtu.pytest_mark_if_available('multiaccelerator')
@jtu.with_config(jax_legacy_prng_key="allow")
class NamedAutodiffTests(jtu.JaxTestCase):

  def testVjpReduceAxes(self):
    raise SkipTest("reduce_axes on vjp is deprecated")
    def f(w, x):
      return jnp.sin(jnp.dot(x, w))

    def vjp_f(w, x, gy):
      _, pullback = jax.vjp(f, w, x)
      return pullback(gy)

    def vjp_f_reduced(w, x, gy):
      _, pullback = jax.vjp(f, w, x, reduce_axes=('batch',))
      return pullback(gy)

    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    gy = np.arange(8, dtype=np.float32).reshape(2, 4)

    # per-example
    error = (r"One of xmap results has an out_axes specification of {}, but is "
             r"actually mapped along more axes defined by this xmap call: "
             r"batch")
    with self.assertRaisesRegex(TypeError, error):
      xmap(vjp_f,
           in_axes=({}, {0: 'batch'}, {0: 'batch'}),
           out_axes=({}, {0: 'batch'}))(w, x, gy)
    out = xmap(vjp_f,
               in_axes=({}, {0: 'batch'}, {0: 'batch'}),
               out_axes=({0: 'batch'}, {0: 'batch'}))(w, x, gy)
    expected = vmap(vjp_f, in_axes=(None, 0, 0), out_axes=(0, 0))(w, x, gy)
    self.assertAllClose(out, expected, check_dtypes=True, rtol=0.005)

    # reduced
    out = xmap(vjp_f_reduced,
               in_axes=({}, {0: 'batch'}, {0: 'batch'}),
               out_axes=({}, {0: 'batch'}))(w, x, gy)
    # the reduced VJP is also the VJP when using a positional batch axis
    expected = vjp_f(w, x, gy)
    self.assertAllClose(out, expected, check_dtypes=True, rtol=0.005)

  def testVjpReduceAxesCollective(self):
    raise SkipTest("reduce_axes on vjp is deprecated")

    # lax.psum has the wrong transpose, so test with a corrected version for now
    @functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
    def psum_idrev(x, axis_name: AxisNames | None = None):
      if axis_name is None:
        return x
      return jax.lax.psum(x, axis_name)

    def psum_idrev_fwd(x, axis_name):
      return psum_idrev(x, axis_name), None

    def psum_idrev_bwd(axis_name, res, g):
      del axis_name, res
      return (g,)

    psum_idrev.defvjp(psum_idrev_fwd, psum_idrev_bwd)

    def f_named(w, x):
      return psum_idrev(jnp.sin(jnp.dot(x, w)).sum(), 'batch')

    def f_positional(w, x):
      return jnp.sin(jnp.dot(x, w)).sum()

    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)

    # forward
    out = xmap(f_named, in_axes=({}, {0: 'batch'}), out_axes={})(w, x)
    expected = f_positional(w, x)
    self.assertAllClose(out, expected, check_dtypes=True)

    # gradient
    out = xmap(jax.grad(f_named, (0, 1), reduce_axes=('batch',)),
               in_axes=({}, {0: 'batch'}),
               out_axes=({}, {0: 'batch'}))(w, x)
    expected = jax.grad(f_positional, (0, 1))(w, x)
    self.assertAllClose(out, expected, check_dtypes=True, rtol=0.005)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
