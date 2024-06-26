# Copyright 2021 The JAX Authors.
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

# Helpers for writing JAX filecheck tests.

import jax
import numpy as np

def print_ir(*prototypes):
  def lower(f):
    """Prints the MLIR IR that results from lowering `f`.

    The arguments to `f` are taken to be arrays shaped like `prototypes`.
    """
    inputs = jax.tree.map(np.array, prototypes)
    flat_inputs, _ = jax.tree.flatten(inputs)
    shape_strs = " ".join([f"{x.dtype.name}[{','.join(map(str, x.shape))}]"
                           for x in flat_inputs])
    name = f.func.__name__ if hasattr(f, "func") else f.__name__
    print(f"\nTEST: {name} {shape_strs}")
    print(jax.jit(f).lower(*inputs).compiler_ir())
  return lower
