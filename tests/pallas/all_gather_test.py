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

"""Tests the simple all_gather kernel."""
from __future__ import annotations

from absl.testing import absltest
import jax
from jax import random
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu import all_gather
import jax.numpy as jnp
import numpy as np

try:
  import hypothesis as hp
  import hypothesis.strategies as hps
  CAN_USE_HYPOTHESIS = True
except (ModuleNotFoundError, ImportError):
  CAN_USE_HYPOTHESIS = False


jax.config.parse_flags_with_absl()
P = jax.sharding.PartitionSpec

if CAN_USE_HYPOTHESIS:

  hp.settings.register_profile(
      "deterministic",
      database=None,
      derandomize=True,
      deadline=None,
      max_examples=50,
      print_blob=True,
      verbosity=hp.Verbosity.verbose,
  )
  hp.settings.load_profile("deterministic")


  @hps.composite
  def _array_shapes(draw):
    # TODO(sharadmv, apaszke): enable this on a wider variety of shapes
    valid_shapes = [
        (128, 128),
        (256, 128),
        (256, 512),
        (256, 1024),
        # TODO(sharadmv,apaszke): enable these shapes
        # (256, 129),
        # (129, 128),
        # (64, 64),
        # (1, 1),
    ]
    return draw(hps.sampled_from(valid_shapes))


  @hps.composite
  def _array_dtypes(draw):
    return draw(
        hps.sampled_from([
            jnp.float32,
            jnp.bfloat16,
            jnp.int32,
            # jnp.float16,  # TODO(sharadmv,apaszke): enable float16 all gather
            # jnp.int16,  # TODO(sharadmv,apaszke): enable int16 all gather
            # jnp.int8,  # TODO(sharadmv,apaszke): enable int8 all gather
        ])
    )


  class AllGatherTest(jtu.JaxTestCase):

    def setUp(self):
      if not jtu.test_device_matches(["tpu"]):
        self.skipTest("Need TPU devices")
      if not jtu.is_device_tpu(version=5, variant="e"):
        # TODO(sharadmv,apaszke): expand support to more versions
        self.skipTest("Currently only supported on TPU v5e")

      super().setUp()

    @hp.given(hps.booleans(), _array_shapes(), _array_dtypes())
    def test_all_gather_1d_mesh(self, is_vmem, shape, dtype):
      if jax.device_count() < 2:
        self.skipTest("Need more devices")
      memory_space = pltpu.VMEM if is_vmem else pltpu.ANY
      mesh_shape = (jax.device_count(),)
      mesh = jax.sharding.Mesh(
          mesh_utils.create_device_mesh(mesh_shape, jax.devices()), ["x"]
      )
      leading, *rest = shape
      shape = (mesh.shape["x"] * leading, *rest)
      x = random.normal(random.key(0), shape, dtype=jnp.float32).astype(dtype)
      x_sharded = jax.device_put(x, jax.sharding.NamedSharding(mesh, P("x")))
      y = all_gather.all_gather(x_sharded, mesh=mesh, axis_name="x",
                                memory_space=memory_space)
      np.testing.assert_array_equal(y, x)

    @hp.given(hps.booleans(), _array_shapes(), _array_dtypes(),
              hps.sampled_from(["x", "y"]))
    def test_all_gather_2d_mesh(self, is_vmem, shape, dtype,
                                axis_name):
      if jax.device_count() < 2:
        self.skipTest("Need more devices")
      if jax.device_count() % 2:
        self.skipTest("Need an even number of devices")
      memory_space = pltpu.VMEM if is_vmem else pltpu.ANY
      mesh_shape = (2, jax.device_count() // 2)
      mesh = jax.sharding.Mesh(
          mesh_utils.create_device_mesh(mesh_shape, jax.devices()), ["x", "y"]
      )
      if axis_name == "x":
        sharding = jax.sharding.NamedSharding(mesh, P("x", None))
      else:
        sharding = jax.sharding.NamedSharding(mesh, P("y", None))
      leading, *rest = shape
      shape = (mesh.shape[axis_name] * leading, *rest)
      x = random.normal(random.key(0), shape, dtype=jnp.float32).astype(dtype)
      x_sharded = jax.device_put(x, sharding)
      y = all_gather.all_gather(x_sharded, mesh=mesh, axis_name=axis_name,
                                memory_space=memory_space)
      np.testing.assert_array_equal(y, x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
