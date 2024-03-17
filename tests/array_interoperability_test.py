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
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.lib import xla_extension_version

import numpy as np

numpy_version = jtu.numpy_version()

config.parse_flags_with_absl()

try:
  import cupy
except ImportError:
  cupy = None

try:
  import tensorflow as tf
  tf_version = tuple(
    int(x) for x in tf.version.VERSION.split("-")[0].split("."))
except ImportError:
  tf = None


dlpack_dtypes = sorted(jax.dlpack.SUPPORTED_DTYPES, key=lambda x: x.__name__)

numpy_dtypes = sorted(
    [dt for dt in jax.dlpack.SUPPORTED_DTYPES if dt != jnp.bfloat16],
    key=lambda x: x.__name__)

# NumPy didn't support bool as a dlpack type until 1.25.
if jtu.numpy_version() < (1, 25, 0):
  numpy_dtypes = [dt for dt in numpy_dtypes if dt != jnp.bool_]

cuda_array_interface_dtypes = [dt for dt in dlpack_dtypes if dt != jnp.bfloat16]

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (2, 3, 4)]
empty_array_shapes = []
empty_array_shapes += [(0,), (0, 4), (3, 0),]
nonempty_nonscalar_array_shapes += [(3, 1), (1, 4), (2, 1, 4)]

nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
all_shapes = nonempty_array_shapes + empty_array_shapes

class DLPackTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest(f"DLPack not supported on {jtu.device_under_test()}")

  @jtu.sample_product(
    shape=all_shapes,
    dtype=dlpack_dtypes,
    gpu=[False, True],
  )
  def testJaxRoundTrip(self, shape, dtype, gpu):
    if xb.using_pjrt_c_api():
      self.skipTest("DLPack support is incomplete in the PJRT C API")  # TODO(skyewm)
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    if gpu and jtu.test_device_matches(["cpu"]):
      raise unittest.SkipTest("Skipping GPU test case on CPU")
    device = jax.devices("gpu" if gpu else "cpu")[0]
    x = jax.device_put(np, device)
    dlpack = jax.dlpack.to_dlpack(x)
    y = jax.dlpack.from_dlpack(dlpack)
    self.assertEqual(y.devices(), {device})
    self.assertAllClose(np.astype(x.dtype), y)

    self.assertRaisesRegex(RuntimeError,
                           "DLPack tensor may be consumed at most once",
                           lambda: jax.dlpack.from_dlpack(dlpack))

  @jtu.sample_product(
    shape=all_shapes,
    dtype=dlpack_dtypes,
    gpu=[False, True],
  )
  def testJaxArrayRoundTrip(self, shape, dtype, gpu):
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    if gpu and jax.default_backend() == "cpu":
      raise unittest.SkipTest("Skipping GPU test case on CPU")
    device = jax.devices("gpu" if gpu else "cpu")[0]
    x = jax.device_put(np, device)
    y = jax.dlpack.from_dlpack(x)
    self.assertEqual(y.devices(), {device})
    self.assertAllClose(np.astype(x.dtype), y)
    # Test we can create multiple arrays
    z = jax.dlpack.from_dlpack(x)
    self.assertEqual(z.devices(), {device})
    self.assertAllClose(np.astype(x.dtype), z)

  @jtu.sample_product(
    shape=all_shapes,
    dtype=dlpack_dtypes,
  )
  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testTensorFlowToJax(self, shape, dtype):
    if xb.using_pjrt_c_api():
      self.skipTest("DLPack support is incomplete in the PJRT C API")
    if (not config.enable_x64.value and
        dtype in [jnp.int64, jnp.uint64, jnp.float64]):
      raise self.skipTest("x64 types are disabled by jax_enable_x64")
    if (jtu.test_device_matches(["gpu"]) and
        not tf.config.list_physical_devices("GPU")):
      raise self.skipTest("TensorFlow not configured with GPU support")

    if jtu.test_device_matches(["gpu"]) and dtype == jnp.int32:
      raise self.skipTest("TensorFlow does not place int32 tensors on GPU")

    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    with tf.device("/GPU:0" if jtu.test_device_matches(["gpu"]) else "/CPU:0"):
      x = tf.identity(tf.constant(np))
    dlpack = tf.experimental.dlpack.to_dlpack(x)
    y = jax.dlpack.from_dlpack(dlpack)
    self.assertAllClose(np, y)

  @jtu.sample_product(
    shape=all_shapes,
    dtype=dlpack_dtypes,
  )
  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testJaxToTensorFlow(self, shape, dtype):
    if (not config.enable_x64.value and
        dtype in [jnp.int64, jnp.uint64, jnp.float64]):
      self.skipTest("x64 types are disabled by jax_enable_x64")
    if (jtu.test_device_matches(["gpu"]) and
        not tf.config.list_physical_devices("GPU")):
      raise self.skipTest("TensorFlow not configured with GPU support")
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    x = jnp.array(np)
    # TODO(b/171320191): this line works around a missing context initialization
    # bug in TensorFlow.
    _ = tf.add(1, 1)
    dlpack = jax.dlpack.to_dlpack(x)
    y = tf.experimental.dlpack.from_dlpack(dlpack)
    self.assertAllClose(np, y.numpy())

  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testTensorFlowToJaxInt64(self):
    if xb.using_pjrt_c_api():
      self.skipTest("DLPack support is incomplete in the PJRT C API")
    # See https://github.com/google/jax/issues/11895
    x = jax.dlpack.from_dlpack(
        tf.experimental.dlpack.to_dlpack(tf.ones((2, 3), tf.int64)))
    dtype_expected = jnp.int64 if config.enable_x64.value else jnp.int32
    self.assertEqual(x.dtype, dtype_expected)

  @jtu.sample_product(
    shape=all_shapes,
    dtype=numpy_dtypes,
  )
  def testNumpyToJax(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x_np = rng(shape, dtype)
    x_jax = jnp.from_dlpack(x_np)
    self.assertAllClose(x_np, x_jax)

  @jtu.sample_product(
    shape=all_shapes,
    dtype=numpy_dtypes,
  )
  @jtu.run_on_devices("cpu") # NumPy only accepts cpu DLPacks
  def testJaxToNumpy(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x_jax = jnp.array(rng(shape, dtype))
    x_np = np.from_dlpack(x_jax)
    self.assertAllClose(x_np, x_jax)

  @unittest.skipIf(xla_extension_version < 221, "Requires newer jaxlib")
  def testNondefaultLayout(self):
    # Generate numpy array with nonstandard layout
    a = np.arange(4).reshape(2, 2)
    b = a.T
    with self.assertRaisesRegex(
        RuntimeError,
        r"from_dlpack got array with non-default layout with minor-to-major "
        r"dimensions \(0,1\), expected \(1,0\)"):
      b_jax = jax.dlpack.from_dlpack(b.__dlpack__())


class CudaArrayInterfaceTest(jtu.JaxTestCase):

  @jtu.skip_on_devices("cuda")
  @unittest.skipIf(xla_extension_version < 228, "Requires newer jaxlib")
  def testCudaArrayInterfaceOnNonCudaFails(self):
    x = jnp.arange(5)
    self.assertFalse(hasattr(x, "__cuda_array_interface__"))
    with self.assertRaisesRegex(
        AttributeError,
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.",
    ):
      _ = x.__cuda_array_interface__

  @jtu.run_on_devices("cuda")
  @unittest.skipIf(xla_extension_version < 233, "Requires newer jaxlib")
  def testCudaArrayInterfaceOnShardedArrayFails(self):
    devices = jax.local_devices()
    if len(devices) <= 1:
      raise unittest.SkipTest("Test requires 2 or more devices")
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, P("x"))
    x = jnp.arange(16)
    x = jax.device_put(x, sharding)
    self.assertFalse(hasattr(x, "__cuda_array_interface__"))
    with self.assertRaisesRegex(
        AttributeError,
        "__cuda_array_interface__ is only supported for unsharded arrays.",
    ):
      _ = x.__cuda_array_interface__


  @jtu.sample_product(
    shape=all_shapes,
    dtype=cuda_array_interface_dtypes,
  )
  @jtu.run_on_devices("cuda")
  def testCudaArrayInterfaceWorks(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    y = jnp.array(x)
    z = np.asarray(y)
    a = y.__cuda_array_interface__
    self.assertEqual(shape, a["shape"])
    self.assertEqual(z.__array_interface__["typestr"], a["typestr"])

  @jtu.run_on_devices("cuda")
  @unittest.skipIf(xla_extension_version < 228, "Requires newer jaxlib")
  def testCudaArrayInterfaceBfloat16Fails(self):
    rng = jtu.rand_default(self.rng())
    x = rng((2, 2), jnp.bfloat16)
    y = jnp.array(x)
    with self.assertRaisesRegex(AttributeError, ".*not supported for BF16.*"):
      _ = y.__cuda_array_interface__

  @jtu.sample_product(
    shape=all_shapes,
    dtype=cuda_array_interface_dtypes,
  )
  @unittest.skipIf(not cupy, "Test requires CuPy")
  @jtu.run_on_devices("cuda")
  def testJaxToCuPy(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    y = jnp.array(x)
    z = cupy.asarray(y)
    self.assertEqual(y.__cuda_array_interface__["data"][0],
                     z.__cuda_array_interface__["data"][0])
    self.assertAllClose(x, cupy.asnumpy(z))

  @unittest.skipIf(xla_extension_version < 237, "Requires newer jaxlib")
  @jtu.sample_product(
    shape=all_shapes,
    dtype=jtu.dtypes.supported(cuda_array_interface_dtypes),
  )
  @unittest.skipIf(not cupy, "Test requires CuPy")
  @jtu.run_on_devices("cuda")
  def testCuPyToJax(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    y = cupy.asarray(x)
    z = jnp.array(y, copy=False)  # this conversion uses dlpack protocol
    self.assertEqual(z.dtype, dtype)
    self.assertEqual(y.__cuda_array_interface__["data"][0],
                     z.__cuda_array_interface__["data"][0])
    self.assertAllClose(np.asarray(z), cupy.asnumpy(y))

  @unittest.skipIf(xla_extension_version < 237, "Requires newer jaxlib")
  @jtu.sample_product(
    shape=all_shapes,
    dtype=jtu.dtypes.supported(cuda_array_interface_dtypes),
  )
  @jtu.run_on_devices("cuda")
  def testCaiToJax(self, shape, dtype):
    # TODO(b/324133505) enable this test for PJRT C API
    if xb.using_pjrt_c_api():
      self.skipTest("CUDA Array Interface support is incomplete in the PJRT C API")
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)

    # using device with highest device_id for testing the correctness
    # of detecting the device id from a pointer value
    device = jax.devices('cuda')[-1]
    with jax.default_device(device):
      y = jnp.array(x, dtype=dtype)
    self.assertEqual(y.dtype, dtype)

    # Using a jax array CAI provider support to construct an object
    # that implements the CUDA Array Interface, versions 2 and 3.
    cai = y.__cuda_array_interface__
    stream = tuple(y.devices())[0].get_stream_for_external_ready_events()

    class CAIWithoutStridesV2:
      __cuda_array_interface__ = cai.copy()
      __cuda_array_interface__["version"] = 2
      # CAI version 2 may not define strides and does not define stream
      __cuda_array_interface__.pop("strides", None)
      __cuda_array_interface__.pop("stream", None)

    class CAIWithoutStrides:
      __cuda_array_interface__ = cai.copy()
      __cuda_array_interface__["version"] = 3
      __cuda_array_interface__["strides"] = None
      __cuda_array_interface__["stream"] = None  # default stream

    class CAIWithStrides:
      __cuda_array_interface__ = cai.copy()
      __cuda_array_interface__["version"] = 3
      strides = (dtype.dtype.itemsize,) if shape else ()
      for s in reversed(shape[1:]):
        strides = (strides[0] * s, *strides)
      __cuda_array_interface__['strides'] = strides
      __cuda_array_interface__["stream"] = stream

    for CAIObject in [CAIWithoutStridesV2, CAIWithoutStrides,
                      CAIWithStrides]:
      z = jnp.array(CAIObject(), copy=False)
      self.assertEqual(y.__cuda_array_interface__["data"][0],
                       z.__cuda_array_interface__["data"][0])
      self.assertAllClose(x, z)
      if 0 in shape:
        # the device id detection from a zero pointer value is not
        # possible
        pass
      else:
        self.assertEqual(y.devices(), z.devices())

      z = jnp.array(CAIObject(), copy=True)
      if 0 not in shape:
        self.assertNotEqual(y.__cuda_array_interface__["data"][0],
                            z.__cuda_array_interface__["data"][0])
      self.assertAllClose(x, z)

class Bfloat16Test(jtu.JaxTestCase):

  @unittest.skipIf((not tf or tf_version < (2, 5, 0)),
                   "Test requires TensorFlow 2.5.0 or newer")
  def testJaxAndTfHaveTheSameBfloat16Type(self):
    self.assertEqual(np.dtype(jnp.bfloat16).num,
                     np.dtype(tf.dtypes.bfloat16.as_numpy_dtype).num)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
