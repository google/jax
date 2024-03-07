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
"""Microbenchmarks for mosaic gpu matrix mutliplication."""

import functools
import sys

from absl import app
import google_benchmark as benchmark
from jax._src import config
from jax.experimental.mosaic.gpu.examples import matmul
from jax._src import test_util as jtu
import jax.numpy as jnp

config.update("jax_traceback_filtering", "off")
config.parse_flags_with_absl()

def _params_name(params):
  return ",".join(f"{k}={v}" for k, v in params.items())

def matmul_benchmark(*args):
  def decorator(get_runtimes):
    for test_case in args:

      @benchmark.register(name=f"{get_runtimes.__name__}_{_params_name(test_case)}")
      @benchmark.option.unit(benchmark.kMillisecond)
      @benchmark.option.use_manual_time()
      @benchmark.option.iterations(1)
      @functools.wraps(get_runtimes)
      def wrapper(state, test_case=test_case):
        m, n, k = test_case["m"], test_case["n"], test_case["k"]
        runtime, ref_runtime = get_runtimes(**test_case)
        state.counters["TFlops"] = (
            float(2 * k * m * n) / (runtime / 1e3) / 1e12
        )
        state.counters["jax_TFlops"] = (
            float(2 * k * m * n) / (ref_runtime / 1e3) / 1e12
        )
        state.counters["speedup"] = ref_runtime / runtime
        state.set_iteration_time(runtime / 1e3)

  return decorator


@matmul_benchmark(
    dict(m=55 * 128, n=95 * 128, k=48 * 128, stages=4, tile_m=128),
    dict(m=55 * 128, n=45 * 128, k=48 * 128, stages=4, tile_m=128),
    dict(m=64, n=95 * 128, k=48 * 128, stages=4, tile_m=64),
    dict(m=64, n=45 * 128, k=48 * 128, stages=4, tile_m=64),
)
def bf16_i8_matmul(m, k, n, stages, tile_m):
  # RHS.element_size==1b so k_tile=128
  if stages * 128 > k:
    raise ValueError(f"Too many stages {(stages, k)=}.")

  return matmul.verify(
      m,
      k,
      n,
      stages,
      tile_m=tile_m,
      rhs_transpose=False,
      lhs_dtype=jnp.bfloat16,
      rhs_dtype=jnp.int8,
  )

@matmul_benchmark(
    dict(m=1024, n=1024, k=1024, stages=4, tile_m=128, tile_n=256),
    dict(m=1024, n=1024, k=1024, stages=4, tile_m=128, tile_n=128),
    dict(m=1024, n=1024, k=1024, stages=4, tile_m=64, tile_n=128),
)
def f32_matmul(m, n, k, stages, tile_m, tile_n):
  if stages * 32 > k:
    raise ValueError(f"Too many stages {(stages, k)=}.")

  return matmul.verify(
      m=m,
      k=k,
      n=n,
      stages=stages,
      tile_m=tile_m,
      tile_n=tile_n,
      rhs_transpose=True,
      lhs_dtype=jnp.float32,
      rhs_dtype=jnp.float32,
  )


def main(_):
  device = jtu.device_under_test()
  if device != "gpu":
    raise ValueError(f"Mosaic only work with gpu (got {device})")

  benchmark.run_benchmarks()


if __name__ == "__main__":
  sys.argv = benchmark.initialize(sys.argv)
  app.run(main)
