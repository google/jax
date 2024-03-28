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

import functools
import math
from absl.testing import absltest
from absl.testing import parameterized
from absl import flags
import unittest

import jax
from jax import lax
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.lib import xla_extension_version
from jax._src import config
from jax.ad_checkpoint import checkpoint_name, checkpoint as new_checkpoint
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.ad_checkpoint import Offloadable, remat
from jax._src.sharding_impls import (NamedSharding, PositionalSharding,
                                     SingleDeviceSharding, GSPMDSharding,
                                     TransferToMemoryKind,
                                     common_devices_indices_map)
import numpy as np

config.parse_flags_with_absl()
FLAGS = flags.FLAGS


def get_memory_kinds_from_executable(f, args):
  compiled = f.lower(*args).compile()
  return compiled.runtime_executable().get_output_memory_kinds()[0]


def _create_inputs(shape, pspec, mem_kind=None):
  mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
  np_inp = np.arange(math.prod(shape)).reshape(shape)
  s = NamedSharding(mesh, pspec, memory_kind=mem_kind)
  inp = jax.device_put(np_inp, s)
  return mesh, s, np_inp, inp


# Tests TODO
# * wsc with memory_kinds
# * shard_map
# * AOT
# * autodiff tests (jtu.check_grads)
# * scan tests
# * jaxpr checks for primitive running on different mem kinds
# * nested jit


class ShardingMemoriesTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Memories do not work on CPU and GPU backends yet.")
    # TODO(b/311021572)
    if jtu.is_cloud_tpu():
      self.skipTest("Experimental feature not yet implemented on Cloud TPU")
    super().setUp()
    self.orig_memories_flag = config.enable_memories.value
    jax.config.update('jax_enable_memories', True)

  def tearDown(self):
    jax.config.update('jax_enable_memories', self.orig_memories_flag)
    super().tearDown()

  @parameterized.named_parameters(
      ("named_sharding", "named_sharding"),
      ("positional_sharding", "positional_sharding"),
      ("single_device_sharding", "single_device_sharding"),
      ("gspmd_sharding", "gspmd_sharding"),
  )
  def test_canonicalize_memory_kind(self, name):
    if name == "named_sharding":
      mesh = jtu.create_global_mesh((1,), "x")
      ns = NamedSharding(mesh, P("x"))
      self.assertEqual(ns.memory_kind, "device")
    elif name == "positional_sharding":
      ps = PositionalSharding(jax.devices())
      self.assertEqual(ps.memory_kind, "device")
    elif name == "single_device_sharding":
      ss = SingleDeviceSharding(jax.devices()[0])
      self.assertEqual(ss.memory_kind, "device")
    else:
      assert name == "gspmd_sharding"
      gs = GSPMDSharding.get_replicated(jax.devices())
      self.assertEqual(gs.memory_kind, "device")

  @parameterized.named_parameters(
      ("named_sharding", "named_sharding"),
      ("positional_sharding", "positional_sharding"),
      ("single_device_sharding", "single_device_sharding"),
      ("gspmd_sharding", "gspmd_sharding"),
  )
  def test_wrong_memory_kind(self, name):
    if name == "named_sharding":
      with self.assertRaisesRegex(
          ValueError, "Could not find memory addressable by device TPU.*"
      ):
        mesh = jtu.create_global_mesh((8,), ("x",))
        NamedSharding(mesh, P("x"), memory_kind="hbm")
    elif name == "positional_sharding":
      with self.assertRaisesRegex(
          ValueError, "Could not find memory addressable by device TPU.*"
      ):
        PositionalSharding(jax.devices(), memory_kind="gpu_hbm")
    elif name == "single_device_sharding":
      with self.assertRaisesRegex(
          ValueError,
          "Could not find memory addressable by device TPU.*Device TPU.*"
          " can address the following memory kinds.*",
      ):
        SingleDeviceSharding(jax.devices()[0], memory_kind="host")
    else:
      assert name == "gspmd_sharding"
      with self.assertRaisesRegex(
          ValueError, "Could not find memory addressable by device TPU.*"
      ):
        GSPMDSharding.get_replicated(jax.devices(), memory_kind="my_host")

  @parameterized.named_parameters(
      ("named_sharding", "named_sharding"),
      ("positional_sharding", "positional_sharding"),
      ("single_device_sharding", "single_device_sharding"),
      ("gspmd_sharding", "gspmd_sharding"),
  )
  def test_correct_tpu_memory_kind(self, name):
    if name == "named_sharding":
      mesh = jtu.create_global_mesh((8,), ("x",))
      NamedSharding(mesh, P("x"), memory_kind="device")
    elif name == "positional_sharding":
      PositionalSharding(jax.devices(), memory_kind="device")
    elif name == "single_device_sharding":
      SingleDeviceSharding(jax.devices()[0], memory_kind="unpinned_host")
    else:
      assert name == "gspmd_sharding"
      GSPMDSharding.get_replicated(jax.devices(), memory_kind="unpinned_host")

  @parameterized.named_parameters(
      ("named_sharding", "named_sharding"),
      ("positional_sharding", "positional_sharding"),
      ("single_device_sharding", "single_device_sharding"),
      ("gspmd_sharding", "gspmd_sharding"),
  )
  def test_sharding_eq(self, name):
    if name == "named_sharding":
      mesh = jtu.create_global_mesh((8,), ("x",))
      s1 = NamedSharding(mesh, P("x"))
      s2 = NamedSharding(mesh, P("x"), memory_kind="device")
      self.assertEqual(s1, s2)
    elif name == "positional_sharding":
      s1 = PositionalSharding(jax.devices())
      s2 = PositionalSharding(jax.devices(), memory_kind="device")
      self.assertEqual(s1, s2)
    elif name == "single_device_sharding":
      s1 = SingleDeviceSharding(jax.devices()[0])
      s2 = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
      self.assertEqual(s1, s2)
    elif name == "gspmd_sharding":
      s1 = GSPMDSharding.get_replicated(jax.devices())
      s2 = GSPMDSharding.get_replicated(jax.devices(), memory_kind="device")
      self.assertEqual(s1, s2)

  def test_sharding_equivalent(self):
    mesh = jtu.create_global_mesh((8,), ("x",))
    ndim = 2
    ns1 = NamedSharding(mesh, P("x"))
    gs1 = GSPMDSharding(
        tuple(mesh.devices.flat),
        ns1._to_xla_hlo_sharding(ndim),
        memory_kind="device",
    )
    self.assertTrue(ns1.is_equivalent_to(gs1, ndim))

    ns2 = NamedSharding(mesh, P("x"), memory_kind="device")
    gs2 = GSPMDSharding(
        tuple(mesh.devices.flat), ns2._to_xla_hlo_sharding(ndim)
    )
    self.assertTrue(ns2.is_equivalent_to(gs2, ndim))

  def test_default_memory_kind(self):
    dev = jax.devices()[0]
    self.assertEqual(dev.default_memory().kind, "device")


class MemoriesComputationTest(jtu.BufferDonationTestCase):

  def setUp(self):
    self.skipTest("Compute via memories does not work yet.")
    super().setUp()

  def _check_mem_kind(self, executable_kind, out_sharding, expected_kind):
    out_kind = out_sharding.memory_kind
    self.assertEqual(executable_kind, out_kind)
    self.assertEqual(out_kind, expected_kind)
    self.assertEqual(executable_kind, expected_kind)

  def _check_device_put_addressable_shards(
      self, out, inp, expected_sharding, expected_mem_kind, index=True):
    self.assertArraysEqual(out, inp)
    self.assertEqual(out.sharding, expected_sharding)
    self.assertEqual(out.sharding.memory_kind, expected_mem_kind)
    for s in out.addressable_shards:
      if index:
        self.assertArraysEqual(s.data, inp[s.index])
      else:
        self.assertArraysEqual(s.data, inp)
      self.assertEqual(s.data.sharding.memory_kind, expected_mem_kind)

  def test_jit_memory_transfer_to_host_middle(self):
    _, s, np_inp, inp = _create_inputs((8, 2), P("x", "y"), mem_kind="device")

    @jax.jit
    def f(x):
      x = x * 2
      y = jax.device_put(x, s.with_memory_kind("unpinned_host"))
      z = y * 3
      a = jax.device_put(z, s.with_memory_kind("device"))
      return a * 4, a

    out1, out2 = f(inp)
    executable_mk = get_memory_kinds_from_executable(f, [inp])

    self.assertArraysEqual(out1, np_inp * 24)
    self.assertArraysEqual(out2, np_inp * 6)
    self.assertEqual(out1.sharding, s)
    self.assertEqual(out2.sharding, s)
    self._check_mem_kind(executable_mk[0], out1.sharding, "device")
    self._check_mem_kind(executable_mk[1], out2.sharding, "device")

  def test_addressable_shards_mem_kind(self):
    _, s, np_inp, inp = _create_inputs((8, 2), P("x", "y"))

    @jax.jit
    def f(x):
      x = jax.device_put(x, s.with_memory_kind("unpinned_host"))
      return x * 2

    out = f(inp)
    executable_mk = get_memory_kinds_from_executable(f, [inp])

    expected_out = np_inp * 2
    self.assertArraysEqual(out, expected_out)
    self.assertEqual(out.sharding, s.with_memory_kind("unpinned_host"))
    self._check_mem_kind(executable_mk[0], out.sharding, "unpinned_host")
    for s in out.addressable_shards:
      self.assertArraysEqual(s.data, expected_out[s.index])
      self._check_mem_kind(executable_mk[0], s.data.sharding, "unpinned_host")

  def test_jit_host_multi_outputs(self):
    _, s, np_inp, inp = _create_inputs((8, 2), P("x"))

    @jax.jit
    def f(x, y):
      x, y = jnp.sin(x), jnp.cos(y)
      x = jax.device_put(x, s.with_memory_kind("unpinned_host"))
      y = jax.device_put(y, s.with_memory_kind("device"))
      return x, y

    out1, out2 = f(inp, inp)

    self.assertArraysAllClose(out1, np.sin(np_inp))
    self.assertArraysAllClose(out2, np.cos(np_inp))
    self.assertEqual(out1.sharding, s.with_memory_kind("unpinned_host"))
    self.assertEqual(out2.sharding, s.with_memory_kind("device"))

  def test_jit_explicit_device(self):
    _, s, np_inp, inp = _create_inputs((8, 2), P("x"), mem_kind="device")

    @jax.jit
    def f(x):
      return x * 2

    out = f(inp)
    executable_mk = get_memory_kinds_from_executable(f, [inp])
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, np_inp * 2)
    self._check_mem_kind(executable_mk[0], out.sharding, "device")

  def test_same_constant_value_on_different_memories(self):
    _, s, np_inp, inp = _create_inputs((8, 2), P("x", "y"), mem_kind="device")

    @jax.jit
    def f(x):
      x = x * 2
      y = jax.device_put(x, s.with_memory_kind("unpinned_host"))
      z = y * 2
      a = jax.device_put(z, s.with_memory_kind("device"))
      return a * 2, z

    out1, out2 = f(inp)
    executable_mk = get_memory_kinds_from_executable(f, [inp])

    self.assertArraysEqual(out1, np_inp * 8)
    self.assertArraysEqual(out2, np_inp * 4)
    self._check_mem_kind(executable_mk[0], out1.sharding, "device")
    self._check_mem_kind(executable_mk[1], out2.sharding, "unpinned_host")

  def test_jit_out_shardings(self):
    _, s, _, inp = _create_inputs((8, 2), P("x", "y"))

    def _check(fun):
      executable_mk = get_memory_kinds_from_executable(fun, [inp])
      outs = fun(inp)
      for o, m in zip(outs, executable_mk):
        self._check_mem_kind(m, o.sharding, "unpinned_host")
        self.assertEqual(o.sharding, s.with_memory_kind("unpinned_host"))

    @functools.partial(
        jax.jit, out_shardings=s.with_memory_kind("unpinned_host")
    )
    def f(x):
      return x * 2, x * 2

    _check(f)

    @functools.partial(
        jax.jit, out_shardings=s.with_memory_kind("unpinned_host")
    )
    def h(x):
      return x, x * 3

    _check(h)

    @functools.partial(
        jax.jit, out_shardings=s.with_memory_kind("unpinned_host")
    )
    def i(x):
      return x, x

    _check(i)

  def test_jit_out_shardings_single_output(self):
    mesh, _, _, inp = _create_inputs((8, 2), P("x", "y"))
    out_s = NamedSharding(mesh, P(), memory_kind="unpinned_host")

    @functools.partial(jax.jit, out_shardings=out_s)
    def g(x):
      return jnp.sum(x * 2)

    out = g(inp)
    self.assertEqual(out.sharding, out_s)
    executable_mk = get_memory_kinds_from_executable(g, [inp])
    self._check_mem_kind(executable_mk[0], out.sharding, "unpinned_host")

    @jax.jit
    def h(x):
      x = jnp.sum(x * 2)
      out = jax.device_put(x, out_s)
      return out

    out = h(inp)
    self.assertEqual(out.sharding, out_s)
    executable_mk = get_memory_kinds_from_executable(h, [inp])
    self._check_mem_kind(executable_mk[0], out.sharding, "unpinned_host")

  def test_jit_device_put_host_output(self):
    _, s, _, inp = _create_inputs((8, 2), P("x", "y"))

    def _check(fun):
      executable_mk = get_memory_kinds_from_executable(fun, [inp])
      outs = fun(inp)
      for o, m in zip(outs, executable_mk):
        self._check_mem_kind(m, o.sharding, "unpinned_host")
        self.assertEqual(o.sharding, s.with_memory_kind("unpinned_host"))

    @jax.jit
    def f(x):
      x = x * 2
      out = jax.device_put(x, s.with_memory_kind("unpinned_host"))
      return out, out

    _check(f)

    @jax.jit
    def h(x):
      x = x * 2
      out = jax.device_put(x, s.with_memory_kind("unpinned_host"))
      return out, out * 3

    _check(h)

    @jax.jit
    def i(x):
      x = x * 2
      out = jax.device_put(x, s.with_memory_kind("unpinned_host"))
      return out * 2, out * 2

    _check(i)

  def test_jit_in_shardings(self):
    _, s, np_inp, inp = _create_inputs((8, 2), P("x", "y"))

    @functools.partial(
        jax.jit, in_shardings=s.with_memory_kind("unpinned_host")
    )
    def f(x):
      return x * 2

    with self.assertRaisesRegex(
        ValueError,
        "Memory kinds passed to jax.jit does not match memory kind on the"
        " respective arg. Got pjit memory kind: unpinned_host, arg memory kind:"
        " device for arg shape.*",
    ):
      f(jnp.arange(16).reshape(8, 2))  # uncommitted inp also raises error

    with self.assertRaisesRegex(
        ValueError,
        "Memory kinds passed to jax.jit does not match memory kind on the"
        " respective arg. Got pjit memory kind: unpinned_host, arg memory kind:"
        " device for arg shape.*",
    ):
      f(inp)  # committed inp raises error.

    @functools.partial(jax.jit, in_shardings=s.with_memory_kind("device"))
    def g(x):
      return x * 2

    out = g(inp)
    executable_kind = get_memory_kinds_from_executable(g, [inp])
    self.assertArraysEqual(out, np_inp * 2)
    self._check_mem_kind(executable_kind[0], out.sharding, "device")

  def test_jit_in_out_shardings(self):
    mesh, s, np_inp, inp = _create_inputs(
        (8, 2), P("x", "y"), mem_kind="device"
    )
    out_s = NamedSharding(mesh, P(), memory_kind="device")

    @functools.partial(jax.jit, in_shardings=s, out_shardings=out_s)
    def f(x):
      return jnp.sum(x)

    out = f(inp)
    executable_kind = get_memory_kinds_from_executable(f, [inp])
    self.assertArraysEqual(out, np.sum(np_inp))
    self._check_mem_kind(executable_kind[0], out.sharding, "device")

    @functools.partial(
        jax.jit,
        in_shardings=s,
        out_shardings=out_s.with_memory_kind("unpinned_host"),
    )
    def g(x):
      return jnp.sum(x)

    out = g(inp)
    executable_kind = get_memory_kinds_from_executable(g, [inp])
    self.assertArraysEqual(out, np.sum(np_inp))
    self._check_mem_kind(executable_kind[0], out.sharding, "unpinned_host")

  def test_device_put_different_devices(self):
    _, _, _, inp = _create_inputs((8, 2), P("x", "y"))

    @jax.jit
    def f(x):
      return jax.device_put(
          x, SingleDeviceSharding(jax.devices()[0], memory_kind="unpinned_host")
      )

    with self.assertRaisesRegex(
        ValueError, "Received incompatible devices for jitted computation"
    ):
      f(inp)

  def test_jit_multiple_transfers(self):
    mesh, _, np_inp, inp = _create_inputs((8, 2), P(None, "y"))
    s2 = NamedSharding(mesh, P("x"))
    inp2 = jax.device_put(np_inp, s2)

    @jax.jit
    def f(x, y):
      a = x + y
      b, c = jax.device_put((a, x), s2.with_memory_kind("unpinned_host"))
      return b * c, y * 2

    out1, out2 = f(inp, inp2)
    executable_mem = get_memory_kinds_from_executable(f, [inp, inp2])
    self.assertArraysEqual(out1, (np_inp + np_inp) * np_inp)
    self.assertArraysEqual(out2, np_inp * 2)
    self._check_mem_kind(executable_mem[0], out1.sharding, "unpinned_host")
    self._check_mem_kind(executable_mem[1], out2.sharding, "device")

  def test_jit_single_device_multi_output_host_mem(self):
    if xb.using_pjrt_c_api():
      raise unittest.SkipTest("GetOutputShardings not supported in PJRT C API")
    inp = jnp.arange(16).reshape(8, 2)

    @jax.jit
    def f(x):
      x = jax.device_put(
          x, SingleDeviceSharding(jax.devices()[0], memory_kind="unpinned_host")
      )
      return x * 2, x * 3

    out1, out2 = f(inp)
    executable_mem = get_memory_kinds_from_executable(f, [inp])
    self.assertArraysEqual(out1, inp * 2)
    self.assertArraysEqual(out2, inp * 3)
    self._check_mem_kind(executable_mem[0], out1.sharding, "unpinned_host")
    self._check_mem_kind(executable_mem[1], out2.sharding, "unpinned_host")

  def test_jit_reshard(self):
    mesh, _, np_inp, inp = _create_inputs((8, 2), P(None, "y"))
    out_s = NamedSharding(mesh, P(("x", "y")), memory_kind="unpinned_host")

    def _check(fun, inp):
      out = fun(inp)
      self.assertArraysEqual(out, np_inp * 2)
      self.assertEqual(out.sharding, out_s)
      executable_kind = get_memory_kinds_from_executable(fun, [inp])
      self._check_mem_kind(executable_kind[0], out.sharding, "unpinned_host")

    @functools.partial(jax.jit, out_shardings=out_s)
    def f(x):
      return x * 2

    _check(f, inp)

    @jax.jit
    def g(x):
      y = jax.device_put(x, out_s)
      return y * 2

    _check(g, inp)

  def test_jit_cpp_cache_hit(self):
    mesh, _, np_inp, inp = _create_inputs((8, 2), P("x", "y"))
    inp2 = jax.device_put(
        np_inp, NamedSharding(mesh, P("x", "y"), memory_kind="device")
    )

    f = jax.jit(lambda x: x @ x.T)

    with jtu.count_pjit_cpp_cache_miss() as count:
      out = f(inp)
      out2 = f(inp2)
    self.assertEqual(count[0], 1)

    self.assertArraysEqual(out, np_inp @ np_inp.T)
    self.assertArraysEqual(out2, np_inp @ np_inp.T)

  def test_jit_compilation_cache_hit(self):
    mesh, s, np_inp, inp = _create_inputs((8, 2), P("x", "y"))
    inp2 = jax.device_put(
        np_inp,
        GSPMDSharding(
            tuple(mesh.devices.flat),
            s._to_xla_hlo_sharding(inp.ndim),
            memory_kind="device",
        ),
    )

    f = jax.jit(lambda x: x @ x.T)

    with (
        jtu.count_pjit_cpp_cache_miss() as cpp_count,
        jtu.count_jit_and_pmap_compiles() as compile_count,
    ):
      f(inp)
      f(inp2)
    self.assertEqual(cpp_count[0], 2)
    self.assertEqual(compile_count[0], 1)

  def test_jit_cpp_cache_output_hit(self):
    _, _, _, inp = _create_inputs((8, 2), P("x"), mem_kind="device")

    @jax.jit
    def mul_two(x):
      return x * 2

    with jtu.count_pjit_cpp_cache_miss() as count:
      out = mul_two(inp)
      mul_two(out)
    self.assertEqual(count[0], 1)

  def test_jit_cache_miss(self):
    mesh, _, np_inp, inp = _create_inputs(
        (8, 2), P("x", "y"), mem_kind="device"
    )
    out_s_host = NamedSharding(mesh, P("x", "y"), memory_kind="unpinned_host")

    @functools.partial(jax.jit, out_shardings=out_s_host)
    def mul_three(x):
      return x * 3

    with (
        jtu.count_pjit_cpp_cache_miss() as cpp_count,
        jtu.count_jit_and_pmap_compiles() as compile_count,
    ):
      out = mul_three(inp)
      out2 = mul_three(out)

    self.assertEqual(cpp_count[0], 2)
    self.assertEqual(compile_count[0], 2)
    self.assertEqual(out.sharding, out_s_host)
    self.assertEqual(out2.sharding, out_s_host)
    self.assertArraysEqual(out, np_inp * 3)
    self.assertArraysEqual(out2, np_inp * 9)
    executable_mk = get_memory_kinds_from_executable(mul_three, [inp])
    self._check_mem_kind(executable_mk[0], out.sharding, "unpinned_host")
    executable_mk2 = get_memory_kinds_from_executable(mul_three, [out])
    self._check_mem_kind(executable_mk2[0], out2.sharding, "unpinned_host")

  def test_jit_host_input_from_another_jit_output(self):
    mesh, _, np_inp, inp = _create_inputs((8, 2), P("x", "y"))
    out_host_s = jax.sharding.NamedSharding(
        mesh, P("x", "y"), memory_kind="unpinned_host"
    )

    @functools.partial(jax.jit, out_shardings=out_host_s)
    def f(x):
      return x * 2

    out = f(inp)
    self.assertEqual(out.sharding, out_host_s)
    executable_kind = get_memory_kinds_from_executable(f, [inp])
    self._check_mem_kind(executable_kind[0], out.sharding, "unpinned_host")
    self.assertArraysEqual(out, np_inp * 2)

    # Input to `f` is on host memory.
    out2 = f(out)
    self.assertEqual(out2.sharding, out_host_s)
    executable_kind = get_memory_kinds_from_executable(f, [out])
    self._check_mem_kind(executable_kind[0], out2.sharding, "unpinned_host")
    self.assertArraysEqual(out2, np_inp * 4)

    lowered_hlo = f.lower(out).as_text(dialect="hlo")
    self.assertIn('_xla_buffer_placement="arg"', lowered_hlo)

  def test_jit_cache_hit_with_default_and_specified_mem_kind(self):
    _, s, np_inp, _ = _create_inputs((8, 2), P("x", "y"))
    _, s2, np_inp2, _ = _create_inputs((8, 2), P("x", "y"), mem_kind="device")

    def mul(x):
      return x @ x.T

    f = jax.jit(mul, in_shardings=s)
    g = jax.jit(mul, in_shardings=s2)

    with jtu.count_jit_and_pmap_compiles() as count:
      out = f(np_inp)
      out2 = g(np_inp2)
    self.assertEqual(count[0], 1)

    self.assertArraysEqual(out, np_inp @ np_inp.T)
    self.assertArraysEqual(out2, np_inp2 @ np_inp2.T)

  def test_sharding_devices_indices_map_cache_hit(self):
    mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    shape = (8, 2)
    s1 = NamedSharding(mesh, P("x", "y"))
    s2 = NamedSharding(mesh, P("x", "y"), memory_kind="device")

    s1.devices_indices_map(shape)
    cache_info1 = common_devices_indices_map.cache_info()
    s2.devices_indices_map(shape)
    cache_info2 = common_devices_indices_map.cache_info()
    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)

  def test_jit_host_inputs_via_device_put_outside(self):
    mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    s_host = NamedSharding(mesh, P("x", "y"), memory_kind="unpinned_host")
    s_hbm = s_host.with_memory_kind("device")
    inp = jnp.arange(16).reshape(8, 2)
    np_inp = np.arange(16).reshape(8, 2)

    inp_host = jax.device_put(inp, s_host)
    inp_hbm = jax.device_put(inp, s_hbm)

    @jax.jit
    def f(x, y):
      return x * 2, y * 2

    out_host, out_hbm = f(inp_host, inp_hbm)

    self._check_device_put_addressable_shards(
        out_host, np_inp * 2, s_host, "unpinned_host")
    self._check_device_put_addressable_shards(
        out_hbm, np_inp * 2, s_hbm, "device")

  def test_trivial_computation(self):
    if xb.using_pjrt_c_api():
      raise unittest.SkipTest("GetOutputShardings not supported in PJRT C API")
    mesh = jtu.create_global_mesh((2, 1), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)

    s_hbm = NamedSharding(mesh, P("x"))
    inp = jax.device_put(np_inp, s_hbm)
    f = jax.jit(lambda x: x)
    out = f(inp)
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.sharding, s_hbm)

    s_host = NamedSharding(mesh, P(None, "x"), memory_kind="unpinned_host")
    inp = jax.device_put(np_inp, s_host)
    f = jax.jit(lambda x: x)
    out = f(inp)
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.sharding, s_host)

  def test_no_donation_across_memory_kinds(self):
    if xb.using_pjrt_c_api():
      raise unittest.SkipTest("GetOutputShardings not supported in PJRT C API")
    mesh = jtu.create_global_mesh((2, 1), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s_hbm = NamedSharding(mesh, P("x"))
    s_host = s_hbm.with_memory_kind("unpinned_host")
    inp = jax.device_put(np_inp, s_hbm)

    @functools.partial(jax.jit, out_shardings=s_host, donate_argnums=0)
    def f(x):
      return x * 2

    with self.assertWarnsRegex(UserWarning, "Some donated buffers were not usable"):
      f(inp)

    lowered_text = f.lower(inp).as_text("hlo")
    self.assertNotIn("input_output_alias", lowered_text)
    self.assertNotDeleted(inp)

  @parameterized.named_parameters(
      ("hbm_to_host", "device", "unpinned_host"),
      ("host_to_hbm", "unpinned_host", "device")
  )
  def test_device_put_memory_kind_no_sharding(self, inp_mem_kind, out_mem_kind):
    mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P("x", "y"), memory_kind=inp_mem_kind)
    inp = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x @ x.T
      z = jax.device_put(y, TransferToMemoryKind(out_mem_kind))
      return z * 2

    out = f(inp)

    self._check_device_put_addressable_shards(
        out, (np_inp @ np_inp.T) * 2,
        NamedSharding(mesh, P("x"), memory_kind=out_mem_kind),
        out_mem_kind)
    executable_kind = get_memory_kinds_from_executable(f, [inp])
    self._check_mem_kind(executable_kind[0], out.sharding, out_mem_kind)

  @parameterized.named_parameters(
      ("hbm_to_host", "device", "unpinned_host"),
      ("host_to_hbm", "unpinned_host", "device")
  )
  def test_device_put_memory_kind_no_sharding_output(
      self, inp_mem_kind, out_mem_kind):
    mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P("x", "y"), memory_kind=inp_mem_kind)
    inp = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x @ x.T
      return jax.device_put(y, TransferToMemoryKind(out_mem_kind))

    out = f(inp)

    self._check_device_put_addressable_shards(
        out, np_inp @ np_inp.T,
        NamedSharding(mesh, P("x"), memory_kind=out_mem_kind),
        out_mem_kind)
    executable_kind = get_memory_kinds_from_executable(f, [inp])
    self._check_mem_kind(executable_kind[0], out.sharding, out_mem_kind)

  @parameterized.named_parameters(
      ("hbm_to_host", "device", "unpinned_host"),
      ("host_to_hbm", "unpinned_host", "device")
  )
  def test_device_put_memory_kind_no_sharding_input(
      self, inp_mem_kind, out_mem_kind):
    mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P("x", "y"), memory_kind=inp_mem_kind)
    inp = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = jax.device_put(x, TransferToMemoryKind(out_mem_kind))
      return y

    # committed sharded input.
    out = f(inp)
    self.assertTrue(out._committed)
    self._check_device_put_addressable_shards(
        out, np_inp, s.with_memory_kind(out_mem_kind), out_mem_kind)

    s1 = SingleDeviceSharding(jax.devices()[1], memory_kind=inp_mem_kind)
    committed_single_device_inp = jax.device_put(np_inp, s1)
    out2 = f(committed_single_device_inp)
    self.assertTrue(out2._committed)
    self._check_device_put_addressable_shards(
        out2, np_inp, s1.with_memory_kind(out_mem_kind), out_mem_kind)

    @jax.jit
    def g(x):
      y = jax.device_put(x, TransferToMemoryKind(out_mem_kind))
      return y

    # Uncommitted input but output will be committed because of device_put.
    out3 = g(np_inp)
    self.assertTrue(out3._committed)
    self._check_device_put_addressable_shards(
        out3, np_inp,
        SingleDeviceSharding(jax.devices()[0], memory_kind=out_mem_kind),
        out_mem_kind)

    @functools.partial(jax.jit, in_shardings=s)
    def h(x):
      y = jax.device_put(x, TransferToMemoryKind(out_mem_kind))
      return y

    out4 = h(np_inp)
    self.assertTrue(out4._committed)
    self._check_device_put_addressable_shards(
        out4, np_inp, s.with_memory_kind(out_mem_kind), out_mem_kind)

  def test_error_transfer_to_memory_kind_outside_jit(self):
    with self.assertRaisesRegex(
        ValueError,
        "TransferToMemoryKind argument to jax.device_put can only be used"
        " inside jax.jit"):
      jax.device_put(np.arange(16), TransferToMemoryKind("device"))

  def test_single_mem_kind_donation_default_mem_kind(self):
    mesh = jtu.create_global_mesh((2,), "x")
    s = NamedSharding(mesh, P())

    @functools.partial(jax.jit, out_shardings=s, donate_argnums=0)
    def f(inp1):
      return inp1 * 2

    x = jax.device_put(np.arange(16).reshape(8, 2), s)

    f(x)

    lowered_text = f.lower(x).as_text("hlo")
    self.assertIn("input_output_alias", lowered_text)
    self.assertDeleted(x)

  def test_host_offload_in_custom_vjp(self):
    if xb.using_pjrt_c_api():
      raise unittest.SkipTest("GetOutputShardings not supported in PJRT C API")
    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)

    def f_fwd(x):
      y = x * 2
      z = jax.device_put(y, TransferToMemoryKind('unpinned_host'))
      return y, (x, z)

    def f_bwd(res, tx):
      x, z = res
      y = x * 2
      z2 = jax.device_put(y, TransferToMemoryKind('unpinned_host'))
      return ((z == z2).astype(jnp.float32),)

    f.defvjp(f_fwd, f_bwd)
    g = jax.jit(jax.grad(lambda x: f(x).sum()))

    x = jnp.ones(3) * 4
    all_true = jnp.ones(3)
    self.assertArraysEqual(g(x), all_true)

  def test_host_offload_in_custom_vjp_sharded(self):
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    s = NamedSharding(mesh, P('x'))

    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)

    def f_fwd(x):
      y = x * 2
      z = jax.device_put(y, s.with_memory_kind('unpinned_host'))
      return y, (x, z)

    def f_bwd(res, tx):
      x, z = res
      y = x * 2
      z2 = jax.device_put(y, s.with_memory_kind('unpinned_host'))
      return ((z == z2).astype(jnp.float32),)

    f.defvjp(f_fwd, f_bwd)
    g = jax.jit(jax.grad(lambda x: f(x).sum()))

    x = jax.device_put(jnp.ones(4) * 4, s)
    all_true = jnp.ones(4)
    self.assertArraysEqual(g(x), all_true)


class DevicePutTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Memories do not work on CPU and GPU backends yet.")
    super().setUp()
    self.orig_memories_flag = config.enable_memories.value
    jax.config.update('jax_enable_memories', True)

  def tearDown(self):
    jax.config.update('jax_enable_memories', self.orig_memories_flag)
    super().tearDown()

  def _check_device_put_addressable_shards(
      self, out, inp, expected_sharding, expected_mem_kind, index=True):
    self.assertArraysEqual(out, inp)
    self.assertEqual(out.sharding, expected_sharding)
    self.assertEqual(out.sharding.memory_kind, expected_mem_kind)
    for s in out.addressable_shards:
      if index:
        self.assertArraysEqual(s.data, inp[s.index])
      else:
        self.assertArraysEqual(s.data, inp)
      self.assertEqual(s.data.sharding.memory_kind, expected_mem_kind)

  def test_device_put_host_to_hbm(self):
    mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    s_host = NamedSharding(mesh, P("y"), memory_kind="unpinned_host")
    np_inp = np.arange(16).reshape(8, 2)

    out_on_host = jax.device_put(np_inp, s_host)
    self.assertEqual(out_on_host.sharding, s_host)

    s_hbm = s_host.with_memory_kind("device")
    out_on_hbm = jax.device_put(out_on_host, s_hbm)
    self._check_device_put_addressable_shards(
        out_on_hbm, np_inp, s_hbm, "device")

    @jax.jit
    def f(x, y):
      return x * 2 + y

    compiled_f = f.lower(out_on_hbm, out_on_hbm).compile()
    print("parameter memory kinds: ", compiled_f.runtime_executable().get_parameter_memory_kinds())
    print("output memory kinds: ", compiled_f.runtime_executable().get_output_memory_kinds())

  def test_device_put_hbm_to_host(self):
    mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    s_host = NamedSharding(mesh, P("y"), memory_kind="unpinned_host")
    inp = jnp.arange(16).reshape(8, 2)

    out_on_host = jax.device_put(inp, s_host)
    self._check_device_put_addressable_shards(
        out_on_host, inp, s_host, "unpinned_host")

    sharded_inp = jax.device_put(inp, s_host.with_memory_kind("device"))
    sharded_out_on_host = jax.device_put(sharded_inp, s_host)
    self._check_device_put_addressable_shards(
        sharded_out_on_host, sharded_inp, s_host, "unpinned_host")

  def test_device_put_different_device_and_memory_host_to_hbm(self):
    if jax.device_count() < 3:
      raise unittest.SkipTest("Test requires >=3 devices")

    out_host0 = jax.device_put(
        jnp.arange(8),
        SingleDeviceSharding(jax.devices()[0], memory_kind="unpinned_host"))

    dev2 = jax.devices()[2]
    out_hbm1 = jax.device_put(
        out_host0, SingleDeviceSharding(dev2, memory_kind="device"))
    self.assertEqual(out_hbm1.sharding.memory_kind, "device")
    self.assertEqual(out_hbm1.sharding._device, dev2)
    self.assertEqual(out_hbm1.addressable_shards[0].data.sharding._device, dev2)
    self.assertEqual(
        out_hbm1.addressable_shards[0].data.sharding.memory_kind, "device")

  def test_device_put_different_device_and_memory_hbm_to_host(self):
    if jax.device_count() < 3:
      raise unittest.SkipTest("Test requires >=3 devices")

    out_hbm0 = jnp.arange(8)

    dev2 = jax.devices()[2]
    out_host1 = jax.device_put(
        out_hbm0, SingleDeviceSharding(dev2, memory_kind="unpinned_host"))
    self.assertEqual(out_host1.sharding.memory_kind, "unpinned_host")
    self.assertEqual(out_host1.sharding._device, dev2)
    self.assertEqual(out_host1.addressable_shards[0].data.sharding._device,
                     dev2)
    self.assertEqual(
        out_host1.addressable_shards[0].data.sharding.memory_kind,
        "unpinned_host")

  def test_device_put_on_different_device_with_the_same_memory_kind(self):
    if len(jax.devices()) < 2:
      raise unittest.SkipTest("Test requires >=2 devices.")

    np_inp = np.arange(16).reshape(8, 2)

    s_hbm_dev_0 = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
    s_hbm_dev_1 = SingleDeviceSharding(jax.devices()[1], memory_kind="device")
    inp_hbm_dev0 = jax.device_put(np_inp, s_hbm_dev_0)
    out_hbm_dev_1 = jax.device_put(inp_hbm_dev0, s_hbm_dev_1)
    self._check_device_put_addressable_shards(
        out_hbm_dev_1, np_inp, s_hbm_dev_1, "device")

    inp_host_dev0 = jax.device_put(
        np_inp, s_hbm_dev_0.with_memory_kind("unpinned_host"))
    s_host_dev_1 = s_hbm_dev_1.with_memory_kind("unpinned_host")
    out_host_dev_1 = jax.device_put(inp_host_dev0, s_host_dev_1)
    self._check_device_put_addressable_shards(
        out_host_dev_1, np_inp, s_host_dev_1, "unpinned_host")

  # TODO(yashkatariya): Enable this once we can compute on host.
  # def test_device_put_resharding(self):
  #   mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
  #   s_host = NamedSharding(mesh, P("x", "y"), memory_kind="unpinned_host")
  #   s_hbm = s_host.with_memory_kind("device")
  #   np_inp = np.arange(16).reshape(8, 2)

  #   # Reshard single device array on HBM to multi device on host
  #   sds_inp_hbm = jax.device_put(
  #       jnp.arange(16).reshape(8, 2),
  #       SingleDeviceSharding(jax.devices()[0], memory_kind="device"))
  #   # device_put on host
  #   out_sharded_host = jax.device_put(sds_inp_hbm, s_host)
  #   self._check_device_put_addressable_shards(
  #       out_sharded_host, np_inp, s_host, "unpinned_host")

  #   # Reshard single device array on host to multi device on hbm
  #   sds_inp_host = jax.device_put(
  #       jnp.arange(16).reshape(8, 2),
  #       SingleDeviceSharding(jax.devices()[0], memory_kind="unpinned_host"))
  #   # device_put on hbm
  #   out_sharded_hbm = jax.device_put(sds_inp_host, s_hbm)
  #   self._check_device_put_addressable_shards(
  #       out_sharded_hbm, np_inp, s_hbm, "device")

  def test_device_put_numpy_array(self):
    mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s_hbm = NamedSharding(mesh, P(("x", "y")), memory_kind="device")
    s_host = s_hbm.with_memory_kind("unpinned_host")

    out_hbm = jax.device_put(np_inp, s_hbm)
    self._check_device_put_addressable_shards(out_hbm, np_inp, s_hbm, "device")

    out_host = jax.device_put(np_inp, s_host)
    self._check_device_put_addressable_shards(
        out_host, np_inp, s_host, "unpinned_host")

  def test_device_put_numpy_scalar(self):
    np_inp = np.float32(8)
    s_hbm = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
    s_host = s_hbm.with_memory_kind("unpinned_host")

    out_hbm = jax.device_put(np_inp, s_hbm)
    self._check_device_put_addressable_shards(out_hbm, np_inp, s_hbm, "device")

    out_host = jax.device_put(np_inp, s_host)
    self._check_device_put_addressable_shards(
        out_host, np_inp, s_host, "unpinned_host")

  def test_device_put_python_scalar(self):
    py_scalar = float(8)
    s_hbm = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
    s_host = s_hbm.with_memory_kind("unpinned_host")

    out_hbm = jax.device_put(py_scalar, s_hbm)
    self._check_device_put_addressable_shards(
        out_hbm, py_scalar, s_hbm, "device", index=False)

    out_host = jax.device_put(py_scalar, s_host)
    self._check_device_put_addressable_shards(
        out_host, py_scalar, s_host, "unpinned_host", index=False)

  def test_device_put_python_int(self):
    py_inp = 8
    s_hbm = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
    s_host = s_hbm.with_memory_kind("unpinned_host")

    out_hbm = jax.device_put(py_inp, s_hbm)
    self._check_device_put_addressable_shards(
        out_hbm, py_inp, s_hbm, "device", index=False)

    out_host = jax.device_put(py_inp, s_host)
    self._check_device_put_addressable_shards(
        out_host, py_inp, s_host, "unpinned_host", index=False)

  def test_parameter_streaming(self):
    _, s_host, np_inp, inp_host = _create_inputs(
        (8, 2), P("x", "y"), mem_kind="pinned_host")
    s_dev = s_host.with_memory_kind('device')
    inp_dev = jax.device_put(np_inp, s_dev)

    @functools.partial(jax.jit, out_shardings=s_host)
    def f(a, b):
      x = b * 2
      y = jax.device_put(a, s_dev)
      z = x * y
      return z * 4, z

    compiled = f.lower(inp_host, inp_dev).compile()  # doesn't crash
    compiled_text = compiled.as_text()
    self.assertRegex(compiled_text, r"entry_computation_layout=.*S\(5\)}")

    out1, out2 = f(inp_host, inp_dev)
    self._check_device_put_addressable_shards(
        out1, np_inp * np_inp * 8, s_host, 'pinned_host')
    self._check_device_put_addressable_shards(
        out2, np_inp * np_inp * 2, s_host, 'pinned_host')


class ActivationOffloadingTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu", "gpu"]):
      self.skipTest("Memories do not work on CPU backend.")
    super().setUp()
    self.orig_memories_flag = config.enable_memories.value
    jax.config.update('jax_enable_memories', True)

  def tearDown(self):
    jax.config.update('jax_enable_memories', self.orig_memories_flag)
    super().tearDown()

  def test_remat_jaxpr_offloadable(self):
    mesh = jtu.create_global_mesh((2,), ("x",))
    inp = jax.device_put(np.arange(16.), NamedSharding(mesh, P("x")))

    def policy(prim, *avals, **params):
      return Offloadable(src="device", dst="pinned_host")

    @functools.partial(remat, policy=policy)
    def f(x):
      x = jnp.sin(x)
      x = jnp.sin(x)
      x = jnp.sin(x)
      return jnp.sum(x)

    fwd_jaxpr, bwd_jaxpr = jtu.fwd_bwd_jaxprs(f, inp)

    self.assertLen(fwd_jaxpr.out_avals, 4)  # 1 output, 3 offloaded residuals
    fwd_mem_kind_count = str(fwd_jaxpr).count(
        "TransferToMemoryKind(memory_kind='pinned_host')")
    self.assertEqual(fwd_mem_kind_count, 3)

    self.assertLen(bwd_jaxpr.in_avals, 4)  # 3 offloaded residuals, 1 input
    bwd_mem_kind_count = str(bwd_jaxpr).count(
        "TransferToMemoryKind(memory_kind='device')")
    self.assertEqual(bwd_mem_kind_count, 3)

    # Execution test.
    f = jax.jit(jax.grad(f))
    f(inp)  # doesn't crash

    compiled_f = f.lower(inp).compile()

    compiled_text = compiled_f.as_text()
    if compiled_text is not None:
      self.assertIn('S(5)', compiled_text)
      self.assertRegex(compiled_text, r"copy-start.*S\(5\)")
      self.assertRegex(compiled_text, r"copy-done.*S\(5\)")

    compiled_stats = compiled_f.memory_analysis()
    if compiled_stats is not None and jtu.test_device_matches(["tpu"]):
      if xla_extension_version >= 240 and jtu.pjrt_c_api_version_at_least(0, 43):
        self.assertGreater(compiled_stats.host_temp_size_in_bytes, 0)

  def test_remat_scan_jaxpr_offloadable(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Remat scan does not work on GPU backend.")
    mesh = jtu.create_global_mesh((2,), ("x",))
    shape = (256, 128)
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    s = NamedSharding(mesh, P("x"))
    inp = jax.device_put(np_inp, s)

    with self.assertRaisesRegex(
        ValueError, "The names should be exclusive and should not intersect"):
      jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=["y"], names_which_can_be_offloaded=["y", "w"],
          offload_src="device", offload_dst="pinned_host")

    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=["y"], names_which_can_be_offloaded=["z", "w"],
        offload_src='device', offload_dst='pinned_host')

    @functools.partial(remat, policy=policy)
    def f(x):
      def g(ys, _):
        y, _ = ys
        y = checkpoint_name(jnp.sin(y), "y")
        z = checkpoint_name(jnp.sin(y), "z")
        z = jax.lax.with_sharding_constraint(z, s)
        w = checkpoint_name(jnp.sin(z), "w")
        return (w, jnp.sum(w)), None
      _, scan_out = jax.lax.scan(g, (x, np.array(1, dtype=np.float32)), [np_inp])[0]
      return scan_out

    fwd_jaxpr, bwd_jaxpr = jtu.fwd_bwd_jaxprs(f, inp)

    self.assertLen(fwd_jaxpr.out_avals, 5)  # 2 output, 3 offloaded residuals
    fwd_mem_kind_count = str(fwd_jaxpr).count(
        "TransferToMemoryKind(memory_kind='pinned_host')")
    self.assertEqual(fwd_mem_kind_count, 2)

    self.assertLen(bwd_jaxpr.in_avals, 5)  # 3 offloaded residuals, 2 input
    bwd_mem_kind_count = str(bwd_jaxpr).count(
        "TransferToMemoryKind(memory_kind='device')")
    self.assertEqual(bwd_mem_kind_count, 2)

    f = jax.jit(jax.grad(f))
    f(inp)  # doesn't crash

    compiled_f = f.lower(inp).compile()

    compiled_text = compiled_f.as_text()
    if compiled_text is not None:
      self.assertIn('S(5)', compiled_text)
      self.assertNotRegex(compiled_text, r"copy-start.*S\(5\)")
      self.assertNotRegex(compiled_text, r"copy-done.*S\(5\)")

    compiled_stats = compiled_f.memory_analysis()
    if compiled_stats is not None:
      if xla_extension_version >= 240 and jtu.pjrt_c_api_version_at_least(0, 43):
        self.assertGreater(compiled_stats.host_temp_size_in_bytes, 0)

  def test_remat_scan_layout_change_offloadable(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Remat scan does not work on GPU backend.")
    mesh = jtu.create_global_mesh((2,), ("x",))
    shape = (256, 128)
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    s = NamedSharding(mesh, P("x"))
    inp = jax.device_put(np_inp, s)

    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=["y"], names_which_can_be_offloaded=["z", "w"],
        offload_src='device', offload_dst='pinned_host')

    @functools.partial(remat, policy=policy)
    def f(x):
      def g(ys, _):
        y, _ = ys
        y = checkpoint_name(jnp.sin(y), "y")
        z = checkpoint_name(jnp.sin(y), "z")
        z = jax.lax.with_sharding_constraint(z, s)
        z = z.T
        w = checkpoint_name(jnp.sin(z), "w")
        return (w.T, jnp.sum(w)), None
      _, scan_out = jax.lax.scan(g, (x, np.array(1, dtype=np.float32)), [np_inp])[0]
      return scan_out

    f = jax.jit(jax.grad(f))
    f(inp)  # doesn't crash

    compiled_f = f.lower(inp).compile()

    compiled_text = compiled_f.as_text()
    if compiled_text is not None:
      self.assertIn('S(5)', compiled_text)
      self.assertNotRegex(compiled_text, r"copy-start.*S\(5\)")
      self.assertNotRegex(compiled_text, r"copy-done.*S\(5\)")

    compiled_stats = compiled_f.memory_analysis()
    if compiled_stats is not None:
      if xla_extension_version >= 240 and jtu.pjrt_c_api_version_at_least(0, 43):
        self.assertGreater(compiled_stats.host_temp_size_in_bytes, 0)

  def test_remat_checkpoint_dots_with_no_batch_dims(self):
    if not jtu.test_device_matches(["tpu"]) and xla_extension_version < 247:
      self.skipTest("Test requires a newer jaxlib")
    policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(
        "device", "pinned_host")

    @functools.partial(new_checkpoint, policy=policy)
    def f(x):
      x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.sum(x)
      return x

    inp = jnp.ones((2, 2))
    f = jax.jit(jax.grad(f))
    f(inp)  # doesn't crash

    compiled_f = f.lower(inp).compile()

    compiled_text = compiled_f.as_text()
    if compiled_text is not None:
      self.assertIn('S(5)', compiled_text)
      self.assertRegex(compiled_text, r"copy-start.*S\(5\)")
      self.assertRegex(compiled_text, r"copy-done.*S\(5\)")

    compiled_stats = compiled_f.memory_analysis()
    if compiled_stats is not None and jtu.test_device_matches(["tpu"]):
      if xla_extension_version >= 240 and jtu.pjrt_c_api_version_at_least(0, 43):
        self.assertGreater(compiled_stats.host_temp_size_in_bytes, 0)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
