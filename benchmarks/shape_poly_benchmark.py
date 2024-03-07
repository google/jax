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
"""Microbenchmarks for JAX shape polymorphism symbolic expressions."""

import google_benchmark as benchmark

import jax
from jax import core
from jax._src.numpy import lax_numpy
from jax import export

jax.config.parse_flags_with_absl()


@benchmark.register
def parse(state):
  while state:
    export.symbolic_shape("a, b, max(a, b), min(max(a, b), b), "
                          "floordiv(a, 2), mod(b, floordiv(a, 2))")

@benchmark.register
def builder_arith(state):
  a, b, c = export.symbolic_shape("a, b, c")
  while state:
    for _ in range(1000):
      e1 = (a + b // a + c % a + 3)
      e2 = (b // a - a - c % a + 4)
      _ = e1 + e2 + (e1 * e2)

@benchmark.register
def builder_linear_arith(state):
  a, b, c = export.symbolic_shape("a, b, c")
  while state:
    left = [a, 3*a, a + 2*b, a + 3*b + 4*c]
    right = [b, -1*a, a - 2*b, a - 3*b - 4*c]
    for l in left:
      for r in right:
        for l_k in [1, 2, -2]:
          for r_k in [1, 2, -2]:
            comb = l * l_k + r * r_k
            if not isinstance(comb, int):
              _ = comb.leading_term  # Ensure we actually materialize


@benchmark.register
def builder_min_max(state):
  a, b = export.symbolic_shape("a, b")
  while state:
    for _ in range(100):
      a.scope._clear_caches()
      _ = core.max_dim(a, b) + core.min_dim(a, a + b)

@benchmark.register
def load_constraints(state):
  while state:
    export.symbolic_shape(
        "a, b, c",
        constraints=["a >= c",
                     "max(max(a, b), 2) >= max(a, b)"])

@benchmark.register
def inequalities_slice(state):

  a, b = export.symbolic_shape("a, b")
  while state:
    for _ in range(30):
      a.scope._clear_caches()
      start, _, slice_size = lax_numpy._preprocess_slice(slice(2, a, 4), b)
      _ = 0 <= slice_size <= b
      _ = start >= 0
      _ = start + slice_size <= b


if __name__ == "__main__":
  benchmark.main()
