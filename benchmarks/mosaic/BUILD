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

load(
    "//jaxlib:jax.bzl",
    "jax_generate_backend_suites",
    "jax_test",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

DISABLED_BACKENDS = [
    "cpu",
    "tpu",
]

DISABLED_CONFIGS = [
    "gpu",
    "gpu_a100",
    "gpu_p100",
    "gpu_p100_x32",
    "gpu_x32",
    "gpu_pjrt_c_api",
]

jax_test(
    name = "matmul_bench",
    srcs = ["matmul_bench.py"],
    disable_backends = DISABLED_BACKENDS,
    disable_configs = DISABLED_CONFIGS,
    tags = ["notap"],
    deps = [
        "//third_party/py/google_benchmark",
        "//third_party/py/jax:mosaic_gpu",
        "//third_party/py/jax/experimental/mosaic/gpu/examples:matmul",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)
