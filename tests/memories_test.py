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

import copy
import functools
import math
import re
from absl.testing import absltest
from absl.testing import parameterized
from absl import flags
import unittest

import jax
from jax import lax
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src import config
from jax.ad_checkpoint import checkpoint_name, checkpoint as new_checkpoint
import jax.numpy as jnp
from jax.ad_checkpoint import Offloadable, remat, Recompute
from jax._src.sharding import common_devices_indices_map
from jax._src.sharding_impls import (NamedSharding, PositionalSharding,
                                     SingleDeviceSharding, GSPMDSharding,
                                     TransferToMemoryKind, PartitionSpec as P)
from jax.experimental.compute_on import compute_on
from jax.experimental.shard_map import shard_map
import numpy as np

config.parse_flags_with_absl()
FLAGS = flags.FLAGS


def get_memory_kinds_from_executable(f, args):
  compiled = f.lower(*args).compile()
  return compiled.runtime_executable().get_output_memory_kinds()[0]


def _create_inputs(shape, pspec, mem_kind=None):
  mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
  np_inp = np.arange(math.prod(shape)).reshape(shape)
  s = NamedSharding(mesh, pspec, memory_kind=mem_kind)
  inp = jax.device_put(np_inp, s)
  return mesh, s, np_inp, inp


@jtu.with_config(jax_enable_memories=True)
class ShardingMemoriesTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["cpu"]):
      self._default_memory_kind = "unpinned_host"
    else:
      self._default_memory_kind = "device"

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
      self.assertEqual(ns.memory_kind, self._default_memory_kind)
    elif name == "positional_sharding":
      ps = PositionalSharding(jax.devices())
      self.assertEqual(ps.memory_kind, self._default_memory_kind)
    elif name == "single_device_sharding":
      ss = SingleDeviceSharding(jax.devices()[0])
      self.assertEqual(ss.memory_kind, self._default_memory_kind)
    else:
      assert name == "gspmd_sharding"
      gs = GSPMDSharding.get_replicated(jax.devices())
      self.assertEqual(gs.memory_kind, self._default_memory_kind)

  @parameterized.named_parameters(
      ("named_sharding", "named_sharding"),
      ("positional_sharding", "positional_sharding"),
      ("single_device_sharding", "single_device_sharding"),
      ("gspmd_sharding", "gspmd_sharding"),
  )
  def test_wrong_memory_kind(self, name):
    if name == "named_sharding":
      with self.assertRaisesRegex(
          ValueError, "Could not find memory addressable by device.*"
      ):
        mesh = jtu.create_global_mesh((1,), ("x",))
        NamedSharding(mesh, P("x"), memory_kind="hbm")
    elif name == "positional_sharding":
      with self.assertRaisesRegex(
          ValueError, "Could not find memory addressable by device.*"
      ):
        PositionalSharding(jax.devices(), memory_kind="gpu_hbm")
    elif name == "single_device_sharding":
      with self.assertRaisesRegex(
          ValueError,
          "Could not find memory addressable by device.*Device.*"
          " can address the following memory kinds.*",
      ):
        SingleDeviceSharding(jax.devices()[0], memory_kind="host")
    else:
      assert name == "gspmd_sharding"
      with self.assertRaisesRegex(
          ValueError, "Could not find memory addressable by device.*"
      ):
        GSPMDSharding.get_replicated(jax.devices(), memory_kind="my_host")

  @parameterized.named_parameters(
      ("named_sharding", "named_sharding"),
      ("positional_sharding", "positional_sharding"),
      ("single_device_sharding", "single_device_sharding"),
      ("gspmd_sharding", "gspmd_sharding"),
  )
  def test_correct_tpu_memory_kind(self, name):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("TPU memory kind test.")

    if name == "named_sharding":
      mesh = jtu.create_global_mesh((1,), ("x",))
      NamedSharding(mesh, P("x"), memory_kind=self._default_memory_kind)
    elif name == "positional_sharding":
      PositionalSharding(jax.devices(), memory_kind=self._default_memory_kind)
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
      mesh = jtu.create_global_mesh((1,), ("x",))
      s1 = NamedSharding(mesh, P("x"))
      s2 = NamedSharding(mesh, P("x"), memory_kind=self._default_memory_kind)
      self.assertEqual(s1, s2)
    elif name == "positional_sharding":
      s1 = PositionalSharding(jax.devices())
      s2 = PositionalSharding(jax.devices(), memory_kind=self._default_memory_kind)
      self.assertEqual(s1, s2)
    elif name == "single_device_sharding":
      s1 = SingleDeviceSharding(jax.devices()[0])
      s2 = SingleDeviceSharding(jax.devices()[0], memory_kind=self._default_memory_kind)
      self.assertEqual(s1, s2)
    elif name == "gspmd_sharding":
      s1 = GSPMDSharding.get_replicated(jax.devices())
      s2 = GSPMDSharding.get_replicated(jax.devices(), memory_kind=self._default_memory_kind)
      self.assertEqual(s1, s2)

  def test_sharding_equivalent(self):
    mesh = jtu.create_global_mesh((1,), ("x",))
    ndim = 2
    ns1 = NamedSharding(mesh, P("x"))
    gs1 = GSPMDSharding(
        tuple(mesh.devices.flat),
        ns1._to_xla_hlo_sharding(ndim),
        memory_kind=self._default_memory_kind,
    )
    self.assertTrue(ns1.is_equivalent_to(gs1, ndim))

    ns2 = NamedSharding(mesh, P("x"), memory_kind=self._default_memory_kind)
    gs2 = GSPMDSharding(
        tuple(mesh.devices.flat), ns2._to_xla_hlo_sharding(ndim)
    )
    self.assertTrue(ns2.is_equivalent_to(gs2, ndim))

  def test_default_memory_kind(self):
    dev = jax.devices()[0]
    self.assertEqual(dev.default_memory().kind, self._default_memory_kind)


@jtu.with_config(jax_enable_memories=True)
class DevicePutTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu", "gpu"]):
      self.skipTest("Memories do not work on CPU backend yet.")
    super().setUp()

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

  def test_error_transfer_to_memory_kind_outside_jit(self):
    with self.assertRaisesRegex(
        ValueError,
        "TransferToMemoryKind argument to jax.device_put can only be used"
        " inside jax.jit"):
      jax.device_put(np.arange(16), TransferToMemoryKind("device"))

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_host_to_hbm(self, host_memory_kind: str):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    s_host = NamedSharding(mesh, P("y"), memory_kind=host_memory_kind)
    np_inp = np.arange(16).reshape(8, 2)

    out_on_host = jax.device_put(np_inp, s_host)
    self.assertEqual(out_on_host.sharding, s_host)

    s_hbm = s_host.with_memory_kind("device")
    out_on_hbm = jax.device_put(out_on_host, s_hbm)
    self._check_device_put_addressable_shards(
        out_on_hbm, np_inp, s_hbm, "device")

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_hbm_to_host(self, host_memory_kind: str):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    s_host = NamedSharding(mesh, P("y"), memory_kind=host_memory_kind)
    inp = jnp.arange(16).reshape(8, 2)

    out_on_host = jax.device_put(inp, s_host)
    self._check_device_put_addressable_shards(
        out_on_host, inp, s_host, host_memory_kind)

    sharded_inp = jax.device_put(inp, s_host.with_memory_kind("device"))
    sharded_out_on_host = jax.device_put(sharded_inp, s_host)
    self._check_device_put_addressable_shards(
        sharded_out_on_host, sharded_inp, s_host, host_memory_kind)

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_different_device_and_memory_host_to_hbm(
      self, host_memory_kind: str
  ):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
    if jax.device_count() < 3:
      raise unittest.SkipTest("Test requires >=3 devices")

    out_host0 = jax.device_put(
        jnp.arange(8),
        SingleDeviceSharding(jax.devices()[0], memory_kind=host_memory_kind))

    dev2 = jax.devices()[2]
    out_hbm1 = jax.device_put(
        out_host0, SingleDeviceSharding(dev2, memory_kind="device"))
    self.assertEqual(out_hbm1.sharding.memory_kind, "device")
    self.assertEqual(out_hbm1.sharding._device, dev2)
    self.assertEqual(out_hbm1.addressable_shards[0].data.sharding._device, dev2)
    self.assertEqual(
        out_hbm1.addressable_shards[0].data.sharding.memory_kind, "device")

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_different_device_and_memory_hbm_to_host(
      self, host_memory_kind: str
  ):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
    if jax.device_count() < 3:
      raise unittest.SkipTest("Test requires >=3 devices")

    out_hbm0 = jnp.arange(8)

    dev2 = jax.devices()[2]
    out_host1 = jax.device_put(
        out_hbm0, SingleDeviceSharding(dev2, memory_kind=host_memory_kind))
    self.assertEqual(out_host1.sharding.memory_kind, host_memory_kind)
    self.assertEqual(out_host1.sharding._device, dev2)
    self.assertEqual(out_host1.addressable_shards[0].data.sharding._device,
                     dev2)
    self.assertEqual(
        out_host1.addressable_shards[0].data.sharding.memory_kind,
        host_memory_kind)

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_on_different_device_with_the_same_memory_kind(
      self, host_memory_kind: str):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
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
        np_inp, s_hbm_dev_0.with_memory_kind(host_memory_kind))
    s_host_dev_1 = s_hbm_dev_1.with_memory_kind(host_memory_kind)
    out_host_dev_1 = jax.device_put(inp_host_dev0, s_host_dev_1)
    self._check_device_put_addressable_shards(
        out_host_dev_1, np_inp, s_host_dev_1, host_memory_kind)

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

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_numpy_array(self, host_memory_kind: str):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s_hbm = NamedSharding(mesh, P(("x", "y")), memory_kind="device")
    s_host = s_hbm.with_memory_kind(host_memory_kind)

    out_hbm = jax.device_put(np_inp, s_hbm)
    self._check_device_put_addressable_shards(out_hbm, np_inp, s_hbm, "device")

    out_host = jax.device_put(np_inp, s_host)
    self._check_device_put_addressable_shards(
        out_host, np_inp, s_host, host_memory_kind)

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_numpy_scalar(self, host_memory_kind: str):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
    np_inp = np.float32(8)
    s_hbm = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
    s_host = s_hbm.with_memory_kind(host_memory_kind)

    out_hbm = jax.device_put(np_inp, s_hbm)
    self._check_device_put_addressable_shards(out_hbm, np_inp, s_hbm, "device")

    out_host = jax.device_put(np_inp, s_host)
    self._check_device_put_addressable_shards(
        out_host, np_inp, s_host, host_memory_kind)

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_python_scalar(self, host_memory_kind: str):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
    py_scalar = float(8)
    s_hbm = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
    s_host = s_hbm.with_memory_kind(host_memory_kind)

    out_hbm = jax.device_put(py_scalar, s_hbm)
    self._check_device_put_addressable_shards(
        out_hbm, py_scalar, s_hbm, "device", index=False)

    out_host = jax.device_put(py_scalar, s_host)
    self._check_device_put_addressable_shards(
        out_host, py_scalar, s_host, host_memory_kind, index=False)

  @parameterized.parameters("unpinned_host", "pinned_host")
  def test_device_put_python_int(self, host_memory_kind: str):
    if jtu.test_device_matches(["gpu"]) and host_memory_kind == "unpinned_host":
      self.skipTest("unpinned_host does not work on GPU backend.")
    py_inp = 8
    s_hbm = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
    s_host = s_hbm.with_memory_kind(host_memory_kind)

    out_hbm = jax.device_put(py_inp, s_hbm)
    self._check_device_put_addressable_shards(
        out_hbm, py_inp, s_hbm, "device", index=False)

    out_host = jax.device_put(py_inp, s_host)
    self._check_device_put_addressable_shards(
        out_host, py_inp, s_host, host_memory_kind, index=False)

  def test_device_put_inside_jit(self):
    _, s_host, np_inp, inp_host = _create_inputs(
        (8, 2), P("x", "y"), mem_kind="pinned_host")
    s_dev = s_host.with_memory_kind("device")

    @jax.jit
    def f(a, b):
      x, y = jax.device_put((a, b), s_dev)
      return x * y

    out = f(inp_host, inp_host)
    self._check_device_put_addressable_shards(
        out, np_inp * np_inp, s_dev, "device")

  def test_parameter_streaming(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test does not work on GPU backend.")
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

  def test_zero_size_parameter(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test does not work on GPU backend.")
    _, s_host, np_inp, inp_host = _create_inputs(
        (0,), P("x"), mem_kind="pinned_host")
    s_dev = s_host.with_memory_kind('device')

    @functools.partial(jax.jit, out_shardings=s_host)
    def f(a):
      b = jax.device_put(a, s_dev)
      return b

    compiled = f.lower(inp_host).compile()  # doesn't crash
    compiled_text = compiled.as_text()
    self.assertRegex(compiled_text, r"entry_computation_layout=.*S\(5\)}")

    out = f(inp_host)
    self._check_device_put_addressable_shards(
        out, np_inp, s_host, 'pinned_host')

  def test_parameter_streaming_with_scalar_and_constant(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test does not work on GPU backend.")
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    scalar_inp = 1
    s_host = NamedSharding(mesh, P(), memory_kind="pinned_host")

    @functools.partial(jax.jit, out_shardings=s_host)
    def f(scalar_input):
      y = jax.device_put(scalar_input, s_host)
      z = 2
      w = jax.device_put(z, s_host)
      return y, w

    compiled = f.lower(scalar_inp).compile()  # doesn't crash
    compiled_text = compiled.as_text()
    self.assertRegex(compiled_text, r"entry_computation_layout=.*S\(5\)}")

    out1, out2 = f(scalar_inp)
    self._check_device_put_addressable_shards(
        out1, scalar_inp, s_host, "pinned_host", index=False
    )
    self._check_device_put_addressable_shards(
        out2, 2, s_host, "pinned_host", index=False
    )

  def test_parameter_and_output_streaming_with_array(self):
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 2:
      self.skipTest("This test requires an xla_version >= 2.")
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s_host = NamedSharding(mesh, P("x", "y"), memory_kind="pinned_host")
    inp_host = jax.device_put(np_inp, s_host)

    @functools.partial(jax.jit, out_shardings=(s_host, s_host))
    def f(x):
      return (x, x)

    compiled = f.lower(inp_host).compile()  # doesn't crash
    compiled_text = compiled.as_text()
    if compiled_text is not None:
      self.assertRegex(compiled_text, r"entry_computation_layout=.*S\(5\)}")

    out1, out2 = f(inp_host)
    self._check_device_put_addressable_shards(
        out1, np_inp, s_host, "pinned_host"
    )
    self._check_device_put_addressable_shards(
        out2, np_inp, s_host, "pinned_host"
    )

  def test_parameter_and_output_streaming_with_scalar(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test is flaky on GPU backend.")
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 2:
      self.skipTest("This test requires an xla_version >= 2.")

    mesh = jax.sharding.Mesh(jax.devices(), "axis")
    s_host = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(), memory_kind="pinned_host"
    )
    scalar_inp = 1

    @functools.partial(jax.jit, out_shardings=(s_host, s_host))
    def f(x):
      return (x, x)

    compiled = f.lower(scalar_inp).compile()  # doesn't crash
    compiled_text = compiled.as_text()
    if compiled_text is not None:
      self.assertRegex(compiled_text, r"entry_computation_layout=.*S\(5\)}")

    out1, out2 = f(scalar_inp)
    self._check_device_put_addressable_shards(
        out1, scalar_inp, s_host, "pinned_host", index=False
    )
    self._check_device_put_addressable_shards(
        out2, scalar_inp, s_host, "pinned_host", index=False
    )

  def test_identity_jit_host_to_device_and_vice_versa(self):
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s_host = NamedSharding(mesh, P('x', 'y'), memory_kind='pinned_host')
    s_dev = s_host.with_memory_kind('device')
    arr_host = jax.device_put(np_inp, s_host)
    arr_dev = jax.device_put(np_inp, s_dev)

    # pinned_host -> device
    f = jax.jit(lambda x: x, out_shardings=s_dev)
    out_dev = f(arr_host)
    self.assertArraysEqual(out_dev, np_inp)
    self.assertEqual(out_dev.sharding, s_dev)

    # device -> pinned_host
    g = jax.jit(lambda x: x, out_shardings=s_host)
    out_host = g(arr_dev)
    self.assertArraysEqual(out_host, np_inp)
    self.assertEqual(out_host.sharding, s_host)

  def test_parameter_streaming_inside_scan(self):
    mesh = jtu.create_global_mesh((1, 1, 2), ("x", "y", "z"))
    np_inp = np.arange(4096.0).reshape(16, 16, 16)
    s_host = NamedSharding(mesh, P("x", "y", "z"), memory_kind="pinned_host")
    arr_host = jax.device_put(np_inp, s_host)

    @jax.jit
    def f(xs):
      def body(carry, x):
        x_tpu = jax.device_put(x, TransferToMemoryKind("device"))
        return carry, x_tpu + carry

      return jax.lax.scan(body, 1.0, xs)

    _, out_hbm = f(arr_host)
    self.assertArraysEqual(out_hbm, np_inp + 1.0)
    # Only expect the last dimension to have a named sharding.
    out_s = NamedSharding(mesh, P(None, None, "z"), memory_kind="device")
    self.assertEqual(out_hbm.sharding, out_s)

  def test_output_streaming(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test is flaky on GPU backend.")
    mesh = jtu.create_global_mesh((1, 1), ("x", "y"))
    np_inp = np.arange(16.0).reshape(8, 2)
    s_hbm = NamedSharding(mesh, P("x", "y"), memory_kind="device")
    s_host = NamedSharding(mesh, P("x", "y"), memory_kind="pinned_host")
    arr_hbm = jax.device_put(np_inp, s_hbm)

    @functools.partial(jax.jit, out_shardings=s_host)
    def f(xs):
      out_tpu = xs + 1.0
      return out_tpu

    out_host = f(arr_hbm)
    self.assertArraysEqual(out_host, np_inp + 1.0)
    self.assertEqual(out_host.sharding, s_host)

  def test_weight_offload_with_dp_on_output(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test is flaky on GPU backend.")
    _, s_dev, np_inp, inp_dev = _create_inputs(
        (8, 2), P("x", "y"), mem_kind="device")
    s_host = s_dev.with_memory_kind('pinned_host')

    @jax.jit
    def f(x):
      x = x * 2
      y = jax.device_put(x, s_host)
      return y

    out_host = f(inp_dev)
    self._check_device_put_addressable_shards(
        out_host, np_inp * 2, s_host, 'pinned_host')

  def test_output_streaming_inside_scan(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test does not work on GPU backend.")
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 2:
      self.skipTest("This test requires an xla_version >= 2.")
    mesh = jtu.create_global_mesh((1, 1, 2), ("x", "y", "z"))
    np_inp = np.arange(4096).reshape(16, 16, 16)
    s_hbm = NamedSharding(mesh, P(None, "y", "z"), memory_kind="device")
    arr_hbm = jax.device_put(np_inp, s_hbm)

    @jax.jit
    def f(xs):
      def body(carry, x):
        out_tpu = x + carry
        return carry, jax.device_put(
            out_tpu, NamedSharding(mesh, P("y", "z"), memory_kind="pinned_host"))
      _, res = jax.lax.scan(body, 1, xs)
      return res

    out = f(arr_hbm)
    self.assertArraysEqual(out, np_inp + 1)
    self.assertEqual(out.sharding.memory_kind, 'pinned_host')

  def test_deepcopy(self):
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 2:
      self.skipTest("This test requires an xla_version >= 2.")
    mesh = jax.sharding.Mesh(jax.devices(), "x")
    s_host = NamedSharding(mesh, P(), memory_kind="pinned_host")

    t = jax.device_put(jnp.zeros((8, 2)), s_host)
    t_copy = copy.deepcopy(t)
    self.assertArraysEqual(t, t_copy)
    self.assertEqual(t.shape, t_copy.shape)

  def test_close_over_host_constant_and_stream(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test does not work on GPU backend.")

    _, s_host, np_inp, inp_host = _create_inputs(
        (8, 2), P("x", "y"), mem_kind="pinned_host")
    s_dev = s_host.with_memory_kind('device')

    @functools.partial(jax.jit, out_shardings=s_dev)
    def f():
      y = jax.device_put(inp_host, s_dev)
      z = y * 2
      return z

    out = f()
    self._check_device_put_addressable_shards(out, np_inp * 2, s_dev, 'device')


@jtu.with_config(jax_enable_memories=True)
class ComputeOffload(jtu.BufferDonationTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Memories do not work on CPU and GPU backends yet.")
    super().setUp()

  def _check_mem_kind(self, executable_kind, out_sharding, expected_kind):
    out_kind = out_sharding.memory_kind
    self.assertEqual(executable_kind, out_kind)
    self.assertEqual(out_kind, expected_kind)
    self.assertEqual(executable_kind, expected_kind)

  def test_compute_no_inputs(self):
    mesh = jtu.create_global_mesh((4,), ('data'))

    tpu_sharding = NamedSharding(mesh, P('data'))
    cpu_sharding = NamedSharding(mesh, P('data'), memory_kind='pinned_host')

    @functools.partial(jax.jit, out_shardings=(tpu_sharding, cpu_sharding))
    def init():
      tpu_array = jax.random.normal(jax.random.key(42), (16,16))
      cpu_array = jax.random.normal(jax.random.key(42), (16,16))
      return tpu_array, cpu_array

    tpu_array, cpu_array = init()
    self.assertEqual(tpu_array.sharding, tpu_sharding)
    self.assertEqual(cpu_array.sharding, cpu_sharding)

  def test_compute_no_inputs_host_replicated(self):
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 3:
      self.skipTest("This test requires an xla_version >= 3.")
    mesh = jtu.create_global_mesh((4,), ('data'))

    tpu_sharding = NamedSharding(mesh, P('data'))
    cpu_sharding = NamedSharding(mesh, P(), memory_kind='pinned_host')

    @functools.partial(jax.jit, out_shardings=(tpu_sharding, cpu_sharding))
    def init():
      tpu_array = jax.random.normal(jax.random.key(42), (16,16))
      cpu_array = jax.random.normal(jax.random.key(42), (16,16))
      return tpu_array, cpu_array

    tpu_array, cpu_array = init()
    self.assertEqual(tpu_array.sharding, tpu_sharding)
    self.assertEqual(cpu_array.sharding, cpu_sharding)

  def test_compute_on_basic(self):
    out_s = SingleDeviceSharding(jax.devices()[0], memory_kind='pinned_host')

    @compute_on('device_host')
    @jax.jit
    def g(x):
      return x * 2

    @jax.jit
    def f(x):
      y = g(x)
      return y * 3

    inp = jnp.arange(8)
    out = f(inp)
    self.assertArraysEqual(out, inp * 6)

    lowered_text = f.lower(jnp.arange(8)).as_text()
    self.assertIn('_xla_compute_type', lowered_text)

    @functools.partial(jax.jit, out_shardings=out_s)
    def h(x):
      y = g(x)
      return y * 3

    out2 = h(inp)
    self.assertArraysEqual(out2, inp * 6)
    self.assertEqual(out2.sharding.memory_kind, 'pinned_host')

  def test_compute_on_reduction(self):
    out_s = SingleDeviceSharding(jax.devices()[0], memory_kind='pinned_host')

    @compute_on('device_host')
    @jax.jit
    def g(x):
      # Reduction generates multiple host computations (inside a single host
      # computation module): the main one and a reduction body.
      return jnp.sum(x)

    @jax.jit
    def f(x):
      y = g(x)
      z = jnp.sum(x)
      return y * z

    inp = jnp.arange(8)
    out = f(inp)
    self.assertArraysEqual(out, np.sum(inp) * np.sum(inp))

    lowered_text = f.lower(jnp.arange(8)).as_text()
    self.assertIn('_xla_compute_type', lowered_text)

    @functools.partial(jax.jit, out_shardings=out_s)
    def h(x):
      y = g(x)
      z = jnp.sum(x)
      return y * z

    out2 = h(inp)
    self.assertArraysEqual(out2, np.sum(inp) * np.sum(inp))
    self.assertEqual(out2.sharding.memory_kind, 'pinned_host')

  def test_nested_compute_error(self):
    @compute_on('device')
    @jax.jit
    def f0(x):
      return x * 2

    @compute_on('device_host')
    @jax.jit
    def f1(x):
      return f0(x)

    @jax.jit
    def f2(x):
      return f1(x)

    with self.assertRaisesRegex(
        NotImplementedError,
        "Nesting `compute_on` with different compute types is not supported"
        " yet."):
      f2(jnp.arange(8))

  def test_compute_on_grad(self):
    @compute_on('device_host')
    @jax.jit
    def g(x):
      return jnp.sin(x)

    def f(x):
      y = g(x)
      return jnp.sum(y)

    inp = jnp.arange(8.)
    jf = jax.jit(jax.grad(f))

    jtu.check_grads(jf, (inp,), order=2)

    lowered_text = jf.lower(inp).as_text('hlo')
    out = re.findall(r"call.*to_apply.*_xla_compute_type", lowered_text)
    self.assertLen(out, 2)

  def test_compute_on_remat(self):
    inp = jnp.arange(16.)

    def policy(prim, *avals, **params):
      return Recompute

    @compute_on('device_host')
    @jax.jit
    def g(x):
      x = jnp.sin(x)
      x = jnp.sin(x)
      x = jnp.sin(x)
      return x

    @functools.partial(remat, policy=policy)
    def f(x):
      x = g(x)
      return jnp.sum(x)

    # Execution test.
    jf = jax.jit(jax.grad(f))
    jf(inp)  # doesn't crash

    lowered_text = jf.lower(inp).as_text('hlo')
    out = re.findall(r"call.*to_apply.*_xla_compute_type", lowered_text)
    self.assertLen(out, 2)

  def test_nested_no_op_compute(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    @compute_on('device_host')
    @jax.jit
    def f0(x):
      return x * 2

    @compute_on('device_host')
    @jax.jit
    def f1(x):
      x = x * 3
      return f0(x)

    @jax.jit
    def f2(x):
      return f1(x)

    out = f2(arr)
    self.assertArraysEqual(out, arr * 6)
    self.assertEqual(out.sharding, s)

  def test_sharded_compute_on_host(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    @compute_on('device_host')
    @jax.jit
    def g(x, y):
      return x * y

    @jax.jit
    def f(x):
      x = x * 3
      return g(x, x)

    out = f(arr)
    expected_out = (np_inp * 3) * (np_inp * 3)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, expected_out)

  def test_host_offload_in_custom_vjp(self):
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 2:
      self.skipTest("This test requires an xla_version >= 2.")
    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)

    @compute_on('device_host')
    @jax.jit
    def eq(x, y):
      return (x == y).astype(jnp.float32)

    def f_fwd(x):
      y = x * 2
      z = jax.device_put(y, TransferToMemoryKind('pinned_host'))
      return y, (x, z)

    def f_bwd(res, tx):
      x, z = res
      y = x * 2
      z2 = jax.device_put(y, TransferToMemoryKind('pinned_host'))
      return (eq(z, z2),)

    f.defvjp(f_fwd, f_bwd)
    g = jax.jit(jax.grad(lambda x: f(x).sum()))

    x = jnp.ones(3) * 4
    all_true = jnp.ones(3)
    self.assertArraysEqual(g(x), all_true)

  def test_host_offload_in_custom_vjp_sharded(self):
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 2:
      self.skipTest("This test requires an xla_version >= 2.")
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    s = NamedSharding(mesh, P('x'))

    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)

    @compute_on('device_host')
    @jax.jit
    def eq(x, y):
      return (x == y).astype(jnp.float32)

    def f_fwd(x):
      y = x * 2
      z = jax.device_put(y, s.with_memory_kind('pinned_host'))
      return y, (x, z)

    def f_bwd(res, tx):
      x, z = res
      y = x * 2
      z2 = jax.device_put(y, s.with_memory_kind('pinned_host'))
      return (eq(z, z2),)

    f.defvjp(f_fwd, f_bwd)
    g = jax.jit(jax.grad(lambda x: f(x).sum()))

    arr = jax.device_put(jnp.ones(4) * 4, s)
    all_true = jnp.ones(4)
    self.assertArraysEqual(g(arr), all_true)

  def test_scan_offload(self):
    np_inp = jnp.arange(4096).reshape(16, 16, 16)

    @jax.jit
    def f(xs):
      def body(carry, x):
        with compute_on('device_host'):
          out_tpu = x + carry
        return carry, out_tpu
      _, res = jax.lax.scan(body, 1, xs)
      return res

    out = f(np_inp)
    self.assertArraysEqual(out, np_inp + 1)

    @compute_on('device_host')
    @jax.jit
    def body2(carry, x):
      out_tpu = x + carry
      return carry, out_tpu

    @jax.jit
    def f2(xs):
      _, res = jax.lax.scan(body2, 1, xs)
      return res

    out2 = f2(np_inp)
    self.assertArraysEqual(out2, np_inp + 1)

  @parameterized.parameters(True, False)
  def test_copy_offload(self, jit_compute_fn: bool):
    # test an explicit copy within the host computation.

    def g(x):
      return jnp.copy(x) * 2

    @jax.jit
    def f(x):
      if jit_compute_fn:
        y = compute_on("device_host")(jax.jit(g))(x)
      else:
        y = compute_on("device_host")(g)(x)
      return y * 3

    inp = jnp.arange(8)
    out = f(inp)
    self.assertArraysEqual(out, inp * 6)

    lowered_text = f.lower(jnp.arange(8)).as_text()
    self.assertIn('_xla_compute_type', lowered_text)

  def test_pure_host_data_and_compute(self):
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 2:
      self.skipTest("This test requires an xla_version >= 2.")
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'), memory_kind='pinned_host')
    np_inp = np.arange(16).reshape(8, 2)
    arr_host = jax.device_put(np_inp, s)

    @compute_on('device_host')
    @jax.jit
    def g(x):
      return x * x

    @functools.partial(jax.jit, out_shardings=s)
    def f(x):
      return g(x)

    out = f(arr_host)
    self.assertEqual(out.sharding, s)
    self.assertEqual(out.sharding.memory_kind, 'pinned_host')
    self.assertArraysEqual(out, np_inp * np_inp)

  def test_eager_compute(self):
    inp = jnp.arange(8.)
    with compute_on('device_host'):
      out = inp * 2
      out = jnp.sin(out)
    self.assertArraysAllClose(out, jnp.sin(inp * 2))

  def test_compute_per_annotation(self):
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    s = NamedSharding(mesh, P("x", "y"))
    np_inp = np.arange(16.).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    @jax.jit
    @compute_on('device_host')
    def f(x):
      return jnp.sin(x * 2)

    # # sharded input
    out = f(arr)
    self.assertArraysAllClose(out, np.sin(np_inp * 2))

    out2 = f(np_inp)
    self.assertArraysAllClose(out2, np.sin(np_inp * 2))

  def test_jit_host_multi_outputs(self):
    if xb.backend_xla_version() is not None and xb.backend_xla_version() < 2:
      self.skipTest("This test requires an xla_version >= 2.")
    _, s, np_inp, inp = _create_inputs((8, 2), P("x"))

    @jax.jit
    def f(x, y):
      x, y = jnp.sin(x), jnp.cos(y)
      x = jax.device_put(x, s.with_memory_kind("pinned_host"))
      y = jax.device_put(y, s.with_memory_kind("device"))
      return x, y

    out1, out2 = f(inp, inp)

    self.assertArraysAllClose(out1, np.sin(np_inp))
    self.assertArraysAllClose(out2, np.cos(np_inp))
    self.assertEqual(out1.sharding, s.with_memory_kind("pinned_host"))
    self.assertEqual(out2.sharding, s.with_memory_kind("device"))

  def test_jit_out_shardings_single_output(self):
    mesh, _, _, inp = _create_inputs((8, 2), P("x", "y"))
    out_s = NamedSharding(mesh, P(), memory_kind="pinned_host")

    @functools.partial(jax.jit, out_shardings=out_s)
    def g(x):
      return jnp.sum(x * 2)

    out = g(inp)
    self.assertEqual(out.sharding, out_s)
    executable_mk = get_memory_kinds_from_executable(g, [inp])
    self._check_mem_kind(executable_mk[0], out.sharding, "pinned_host")

    @jax.jit
    def h(x):
      x = jnp.sum(x * 2)
      out = jax.device_put(x, out_s)
      return out

    out = h(inp)
    self.assertEqual(out.sharding, out_s)
    executable_mk = get_memory_kinds_from_executable(h, [inp])
    self._check_mem_kind(executable_mk[0], out.sharding, "pinned_host")

  def test_jit_in_shardings(self):
    _, s, np_inp, inp = _create_inputs((8, 2), P("x", "y"))

    @functools.partial(jax.jit, in_shardings=s.with_memory_kind("pinned_host"))
    def f(x):
      return x * 2

    with self.assertRaisesRegex(
        ValueError,
        "Memory kinds passed to jax.jit does not match memory kind on the"
        " respective arg. Got pjit memory kind: pinned_host, arg memory kind:"
        " device for arg shape.*"):
      f(jnp.arange(16).reshape(8, 2))  # uncommitted inp also raises error

    with self.assertRaisesRegex(
        ValueError,
        "Memory kinds passed to jax.jit does not match memory kind on the"
        " respective arg. Got pjit memory kind: pinned_host, arg memory kind:"
        " device for arg shape.*"):
      f(inp)  # committed inp raises error.

    @functools.partial(jax.jit, in_shardings=s.with_memory_kind("device"))
    def g(x):
      return x * 2

    out = g(inp)
    executable_kind = get_memory_kinds_from_executable(g, [inp])
    self.assertArraysEqual(out, np_inp * 2)
    self._check_mem_kind(executable_kind[0], out.sharding, "device")

  def test_jit_in_out_shardings(self):
    mesh, s, np_inp, inp = _create_inputs((8, 2), P("x", "y"), mem_kind="device")
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
        out_shardings=out_s.with_memory_kind("pinned_host"),
    )
    def g(x):
      return jnp.sum(x)

    out = g(inp)
    executable_kind = get_memory_kinds_from_executable(g, [inp])
    self.assertArraysEqual(out, np.sum(np_inp))
    self._check_mem_kind(executable_kind[0], out.sharding, "pinned_host")

  def test_device_put_different_devices(self):
    _, _, _, inp = _create_inputs((8, 2), P("x", "y"))

    @jax.jit
    def f(x):
      return jax.device_put(
          x, SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host"))

    with self.assertRaisesRegex(
        ValueError, "Received incompatible devices for jitted computation"):
      f(inp)

  def test_jit_cpp_cache_hit(self):
    mesh, _, np_inp, inp = _create_inputs((8, 2), P("x", "y"))
    inp2 = jax.device_put(
        np_inp, NamedSharding(mesh, P("x", "y"), memory_kind="device"))

    f = jax.jit(lambda x: x @ x.T)

    with jtu.count_pjit_cpp_cache_miss() as count:
      out = f(inp)
      out2 = f(inp2)
    self.assertEqual(count[0], 1)

    self.assertArraysEqual(out, np_inp @ np_inp.T)
    self.assertArraysEqual(out2, np_inp @ np_inp.T)

  def test_jit_cpp_cache_output_hit(self):
    _, _, _, inp = _create_inputs((8, 2), P("x"), mem_kind="device")

    @jax.jit
    def mul_two(x):
      return x * 2

    with jtu.count_pjit_cpp_cache_miss() as count:
      out = mul_two(inp)
      mul_two(out)
    self.assertEqual(count[0], 1)

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
    mesh = jtu.create_global_mesh((2, 2), ("x", "y"))
    shape = (8, 2)
    s1 = NamedSharding(mesh, P("x", "y"))
    s2 = NamedSharding(mesh, P("x", "y"), memory_kind="device")

    s1.devices_indices_map(shape)
    cache_info1 = common_devices_indices_map.cache_info()
    s2.devices_indices_map(shape)
    cache_info2 = common_devices_indices_map.cache_info()
    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)

  def test_no_donation_across_memory_kinds(self):
    if xb.using_pjrt_c_api():
      raise unittest.SkipTest("GetOutputShardings not supported in PJRT C API")
    mesh = jtu.create_global_mesh((2, 1), ("x", "y"))
    np_inp = np.arange(16).reshape(8, 2)
    s_hbm = NamedSharding(mesh, P("x"))
    s_host = s_hbm.with_memory_kind("pinned_host")
    inp = jax.device_put(np_inp, s_hbm)

    @functools.partial(jax.jit, out_shardings=s_host, donate_argnums=0)
    def f(x):
      return x * 2

    with self.assertWarnsRegex(
        UserWarning, "Some donated buffers were not usable"):
      f(inp)

    lowered_text = f.lower(inp).as_text("hlo")
    self.assertNotIn("input_output_alias", lowered_text)
    self.assertNotDeleted(inp)

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

  def test_compute_offload_inside_shmap(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    @compute_on('device_host')
    @jax.jit
    def g(x):
      return x * 2

    def f(x):
      x = x * 3
      y = g(x)
      return y * 4

    out = jax.jit(shard_map(f, mesh=mesh, in_specs=P('x', 'y'),
                            out_specs=P('x', 'y')))(arr)
    self.assertArraysEqual(out, np_inp * 24)

  def test_qr_decomposition_offload(self):
    if jtu.is_cloud_tpu():
      self.skipTest("Test fails on cloud TPU")

    shape = (3, 3)
    dtype = np.float32
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)

    @compute_on("device_host")
    @jax.jit
    def g(x):
      return lax.linalg.qr(x, full_matrices=True)

    @jax.jit
    def f(x):
      x, _ = lax.linalg.qr(x, full_matrices=True)
      x, _ = g(x)
      return x

    out = f(operand)  # doesn't crash
    lowered_text = f.lower(operand).as_text()
    self.assertIn('@lapack_sgeqrf', lowered_text)
    self.assertIn('@Qr', lowered_text)

    @jax.jit
    def h(x):
      x, _ = lax.linalg.qr(x, full_matrices=True)
      x, _ = lax.linalg.qr(x, full_matrices=True)
      return x

    expected_out = h(operand)

    self.assertArraysAllClose(out, expected_out, rtol=1e-3)

  def test_mem_kind_donation_pinned_host(self):
    mesh = jtu.create_global_mesh((2,), "x")
    s = NamedSharding(mesh, P(), memory_kind='pinned_host')
    s_dev = s.with_memory_kind('device')

    @compute_on('device_host')
    @functools.partial(jax.jit, out_shardings=(s, s_dev), donate_argnums=(0, 1))
    def f(inp1, inp2):
      return inp1 * 2, inp2 * 2

    np_inp = np.arange(16).reshape(8, 2)
    x = jax.device_put(np_inp, s)
    x_dev = jax.device_put(np_inp, s_dev)

    f(x, x_dev)

    lowered_text = f.lower(x, x_dev).as_text("hlo")
    self.assertIn("input_output_alias", lowered_text)
    self.assertDeleted(x)
    self.assertDeleted(x_dev)

  @parameterized.parameters("pinned_host", "device")
  def test_identity_mem_kind_donation(self, mem_kind):
    mesh = jtu.create_global_mesh((2,), "x")
    s = NamedSharding(mesh, P(), memory_kind=mem_kind)

    @functools.partial(jax.jit, out_shardings=s, donate_argnums=0)
    def f(inp):
      return inp

    np_inp = np.arange(16).reshape(8, 2)
    x = jax.device_put(np_inp, s)

    f(x)

    lowered_text = f.lower(x).as_text("hlo")
    self.assertIn("input_output_alias", lowered_text)
    self.assertDeleted(x)

  @jtu.run_on_devices('tpu')
  def test_aot_device_implicit_transfer(self):
    mesh = jtu.create_global_mesh((1,), 'x')
    np_inp = np.arange(8)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P()))

    @jax.jit
    def f(x):
      return x * 2

    compiled = f.lower(arr).compile()

    cpu_dev = jax.devices('cpu')[0]
    with jax.default_device(cpu_dev):
      cpu_arr = jnp.arange(8)
      self.assertEqual(cpu_arr.sharding, SingleDeviceSharding(cpu_dev))
      self.assertFalse(cpu_arr._committed)

    out = compiled(cpu_arr)
    self.assertArraysEqual(out, np_inp * 2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P()))
    self.assertEqual(out.sharding.memory_kind, 'device')


@jtu.with_config(jax_enable_memories=True)
class ActivationOffloadingTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu", "gpu"]):
      self.skipTest("Memories do not work on CPU backend.")
    super().setUp()

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
    if compiled_stats is not None:
      if jtu.pjrt_c_api_version_at_least(0, 43):
        self.assertGreater(compiled_stats.host_temp_size_in_bytes, 0)

  def test_remat_scan_jaxpr_offloadable(self):
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
      if jtu.pjrt_c_api_version_at_least(0, 43):
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
      if jtu.pjrt_c_api_version_at_least(0, 43):
        self.assertGreater(compiled_stats.host_temp_size_in_bytes, 0)

  def test_remat_checkpoint_dots_with_no_batch_dims(self):
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
    if compiled_stats is not None:
      if jtu.pjrt_c_api_version_at_least(0, 43):
        self.assertGreater(compiled_stats.host_temp_size_in_bytes, 0)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
