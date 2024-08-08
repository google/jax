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
import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
import jax
from jax import random
from jax._src.lib import xla_extension
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas.pallas_call import _trace_kernel_to_jaxpr
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


# TODO(sharadmv): Update signatures of pallas_call to correct inputs/outputs.
# pylint: disable=no-value-for-parameter

config.parse_flags_with_absl()


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jtu.test_device_matches(["cpu"]) and not self.INTERPRET:
      self.skipTest("On CPU the test works only in interpret mode")
    if jtu.test_device_matches(["gpu"]) and jax.config.x64_enabled:
      self.skipTest("On GPU the test works only in 32-bit")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPU with capability >= sm80")
    if sys.platform == "win32" and not self.INTERPRET:
      self.skipTest("Only works on non-Windows platforms")

    super().setUp()
    _trace_kernel_to_jaxpr.cache_clear()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


class PallasCallVmapTest(PallasBaseTest):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["tpu"]):
      # TODO: most tests fail on TPU in non-interpreter mode
      self.skipTest("On TPU the test works only in interpret mode")

  def test_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
    )
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(add_one)(jnp.arange(8))
    out_ref = jnp.arange(1, 9)
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_simple_kernel_with_in_axes_None(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
    )
    def add(x_ref, y_ref, o_ref):
      o_ref[()] = x_ref[()] + y_ref[()]
    out = jax.vmap(add, in_axes=(0, None))(jnp.arange(8), 1)
    out_ref = jnp.arange(1, 9)
    np.testing.assert_allclose(out, out_ref)

  def test_double_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
    )
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(jax.vmap(add_one))(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_quadruple_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
    )
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(jax.vmap(jax.vmap(jax.vmap(add_one))))(
        jnp.arange(15 * 8).reshape((5, 3, 4, 2)))
    out_ref = jnp.arange(1, 15 * 8 + 1).reshape((5, 3, 4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_quadruple_vmap_of_batched_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((7,), jnp.int32),
        grid=(7,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    out = jax.vmap(jax.vmap(jax.vmap(jax.vmap(add_one))))(
        jnp.arange(15 * 8 * 7).reshape((5, 3, 4, 2, 7)))
    out_ref = jnp.arange(1, 15 * 8 * 7 + 1).reshape((5, 3, 4, 2, 7))
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_slicing_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        grid=(2,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    out = jax.vmap(add_one)(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_with_hoisted_consts(self):
    to_store = np.arange(128, dtype=np.float32).reshape((1, 128))
    x = np.arange(4 * 16 * 128, dtype=np.float32).reshape((4, 16, 128))

    @jax.vmap
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((64, 128), x.dtype),
        grid=(2,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((32, 128), lambda i: (i, 0)),
    )
    def kernel(src, dst):
      dst[0:1] = to_store

    with self.assertRaisesRegex(
        ValueError,
        "The kernel function .* captures constants"):
      kernel(x)

  def test_vmap_of_kernel_with_input_output_aliases(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        input_output_aliases={1:0},
        grid=())
    def add(x_ref, _, o_ref):
      o_ref[()] = x_ref[()] + o_ref[()] + 1
    out = jax.vmap(add, in_axes=(0, None))(jnp.arange(8), 1)
    out_ref = jnp.arange(2, 10)
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_kernel_with_input_output_aliases_different_axes(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.int32),
        input_output_aliases={0: 0},
        grid=(),
    )
    def add(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1

    out = jax.vmap(add, in_axes=1)(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2)).swapaxes(0, 1)
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_slicing_kernel_different_axes(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        grid=(2,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    add_one_ref = lambda x: x + 1
    x = jnp.arange(8).reshape((2, 4))

    out = jax.vmap(add_one, in_axes=1, out_axes=1)(x)
    out_ref = jax.vmap(add_one_ref, in_axes=1, out_axes=1)(x)
    np.testing.assert_allclose(out, out_ref)

    out = jax.vmap(add_one, in_axes=1, out_axes=0)(x)
    out_ref = jax.vmap(add_one_ref, in_axes=1, out_axes=0)(x)
    np.testing.assert_allclose(out, out_ref)

  def test_double_vmap_of_slicing_kernel_different_axes(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        grid=(4,))
    def sin(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = jnp.sin(x_ref[i])
    sin_ref = jnp.sin
    x = jnp.arange(64.).reshape((8, 4, 2))

    out = jax.vmap(jax.vmap(sin, in_axes=1), in_axes=0)(x)
    out_ref = jax.vmap(jax.vmap(sin_ref, in_axes=1), in_axes=0)(x)
    np.testing.assert_allclose(out, out_ref, atol=1e-3, rtol=1e-3)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_small_large_vmap(self):
    if xla_extension.is_tsan() and jtu.test_device_matches(["cpu"]):
      self.skipTest("Test is very slow under TSAN")

    # Catches https://github.com/google/jax/issues/18361
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        grid=(2,))
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1

    add_one = jax.vmap(jax.vmap(add_one))
    add_one_ref = lambda x: x + 1

    x = random.randint(random.key(0), (4, 65536, 2), 0, 10000)

    out = add_one(x)
    out_ref = add_one_ref(x)

    np.testing.assert_allclose(out, out_ref)

  def test_small_small_large_vmap(self):
    if xla_extension.is_tsan() and jtu.test_device_matches(["cpu"]):
      self.skipTest("Test is very slow under TSAN")

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        grid=(2,))
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1

    add_one = jax.vmap(jax.vmap(jax.vmap(add_one)))
    add_one_ref = lambda x: x + 1

    x = random.randint(random.key(0), (2, 2, 65536, 2), 0, 10000)

    out = add_one(x)
    out_ref = add_one_ref(x)

    np.testing.assert_allclose(out, out_ref)


class PallasCallVmapInterpreterTest(PallasCallVmapTest):
  INTERPRET = True

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["cpu"]) and jax.config.x64_enabled:
      # TODO: assertion failures on CPU in 64-bit mode
      self.skipTest("On CPU the test works only in 32-bit mode")


if __name__ == "__main__":
  absltest.main()
