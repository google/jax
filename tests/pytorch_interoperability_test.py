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

import unittest

from absl.testing import absltest

import jax
import jax.dlpack
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.lib import xla_client
import jax.numpy as jnp

config.parse_flags_with_absl()

try:
  import torch
  import torch.utils.dlpack
except ImportError:
  torch = None


torch_dtypes = [jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                jnp.uint8, jnp.float16, jnp.float32, jnp.float64,
                jnp.bfloat16, jnp.complex64, jnp.complex128]

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (2, 3, 4)]
empty_array_shapes = []
empty_array_shapes += [(0,), (0, 4), (3, 0), (2, 0, 1)]
nonempty_nonscalar_array_shapes += [(3, 1), (1, 4), (2, 1, 4)]

nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
all_shapes = nonempty_array_shapes + empty_array_shapes

@unittest.skipIf(not torch, "Test requires PyTorch")
class DLPackTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("DLPack only supported on CPU and GPU")

  def testTorchToJaxFailure(self):
    x = torch.arange(6).reshape((2, 3))
    x = x.cuda() if jtu.test_device_matches(["gpu"]) else x
    y = torch.utils.dlpack.to_dlpack(x[:, :2])

    backend = xla_bridge.get_backend()
    client = getattr(backend, "client", backend)

    regex_str = (r'UNIMPLEMENTED: Only DLPack tensors with trivial \(compact\) '
                 r'striding are supported')
    with self.assertRaisesRegex(RuntimeError, regex_str):
      xla_client._xla.dlpack_managed_tensor_to_buffer(
          y, client, client)

  @jtu.sample_product(shape=all_shapes, dtype=torch_dtypes)
  def testJaxToTorch(self, shape, dtype):
    if not config.enable_x64.value and dtype in [
        jnp.int64,
        jnp.float64,
        jnp.complex128,
    ]:
      self.skipTest("x64 types are disabled by jax_enable_x64")
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    x = jnp.array(np)
    dlpack = jax.dlpack.to_dlpack(x)
    y = torch.utils.dlpack.from_dlpack(dlpack)
    if dtype == jnp.bfloat16:
      # .numpy() doesn't work on Torch bfloat16 tensors.
      self.assertAllClose(np,
                          y.cpu().view(torch.int16).numpy().view(jnp.bfloat16))
    else:
      self.assertAllClose(np, y.cpu().numpy())

  @jtu.sample_product(shape=all_shapes, dtype=torch_dtypes)
  def testJaxArrayToTorch(self, shape, dtype):
    if not config.enable_x64.value and dtype in [
        jnp.int64,
        jnp.float64,
        jnp.complex128,
    ]:
      self.skipTest("x64 types are disabled by jax_enable_x64")
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    # Test across all devices
    for device in jax.local_devices():
      x = jax.device_put(np, device)
      y = torch.utils.dlpack.from_dlpack(x)
      if dtype == jnp.bfloat16:
        # .numpy() doesn't work on Torch bfloat16 tensors.
        self.assertAllClose(
            np, y.cpu().view(torch.int16).numpy().view(jnp.bfloat16)
        )
      else:
        self.assertAllClose(np, y.cpu().numpy())

  def testTorchToJaxInt64(self):
    # See https://github.com/google/jax/issues/11895
    x = jax.dlpack.from_dlpack(
        torch.utils.dlpack.to_dlpack(torch.ones((2, 3), dtype=torch.int64)))
    dtype_expected = jnp.int64 if config.enable_x64.value else jnp.int32
    self.assertEqual(x.dtype, dtype_expected)

  @jtu.sample_product(shape=all_shapes, dtype=torch_dtypes)
  def testTorchToJax(self, shape, dtype):
    if xla_bridge.using_pjrt_c_api():
      self.skipTest("x64 types are disabled by jax_enable_x64")
    if not config.enable_x64.value and dtype in [
        jnp.int64,
        jnp.float64,
        jnp.complex128,
    ]:
      self.skipTest("x64 types are disabled by jax_enable_x64")

    rng = jtu.rand_default(self.rng())
    x_np = rng(shape, dtype)
    if dtype == jnp.bfloat16:
      x = torch.tensor(x_np.view(jnp.int16)).view(torch.bfloat16)
    else:
      x = torch.tensor(x_np)
    x = x.cuda() if jtu.test_device_matches(["gpu"]) else x
    x = x.contiguous()
    y = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    self.assertAllClose(x_np, y)

    # Verify the resulting value can be passed to a jit computation.
    z = jax.jit(lambda x: x + 1)(y)
    self.assertAllClose(x_np + dtype(1), z)

  @jtu.sample_product(shape=all_shapes, dtype=torch_dtypes)
  def testTorchToJaxArray(self, shape, dtype):
    if not config.enable_x64.value and dtype in [
        jnp.int64,
        jnp.float64,
        jnp.complex128,
    ]:
      self.skipTest("x64 types are disabled by jax_enable_x64")

    rng = jtu.rand_default(self.rng())
    x_np = rng(shape, dtype)
    if dtype == jnp.bfloat16:
      x = torch.tensor(x_np.view(jnp.int16)).view(torch.bfloat16)
    else:
      x = torch.tensor(x_np)
    x = x.cuda() if jtu.test_device_matches(["gpu"]) else x
    x = x.contiguous()
    y = jax.dlpack.from_dlpack(x)
    self.assertAllClose(x_np, y)

    # Verify the resulting value can be passed to a jit computation.
    z = jax.jit(lambda x: x + 1)(y)
    self.assertAllClose(x_np + dtype(1), z)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
