# Copyright 2024 The JAX Authors.
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

# ruff: noqa

import datetime
from jax import numpy as jnp
import numpy as np

data_2024_02_20 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['Subchannel'],
    serialized_date=datetime.date(2024, 2, 20),
    inputs=(np.ones((2, 256), dtype=jnp.bfloat16), np.ones((2, 128, 2), dtype=jnp.int4), np.ones((2, 2), dtype=jnp.bfloat16)),
    expected_outputs=(np.ones((2, 2), dtype=jnp.float32)),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main(%arg0: tensor<2x256xbf16>, %arg1: tensor<2x128x2xs4>, %arg2: tensor<128x2xbf16>) -> (tensor<2x2xf32>) {
    %0 = stablehlo.custom_call @Subchannel(%arg0, %arg1, %arg2) : (tensor<2x256xbf16>, tensor<2x128x2xs4>, tensor<128x2xbf16>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
""",
    mlir_module_serialized=b"",
    xla_call_module_version=5,
)  # End paste
