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
    name = "gpu_test",
    srcs = ["gpu_test.py"],
    disable_backends = DISABLED_BACKENDS,
    disable_configs = DISABLED_CONFIGS,
    shard_count = 4,
    deps = [
        "//jax:mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_test(
    name = "matmul_test",
    srcs = ["matmul_test.py"],
    disable_backends = DISABLED_BACKENDS,
    disable_configs = DISABLED_CONFIGS,
    shard_count = 8,
    deps = [
        "//jax:mosaic_gpu",
        "//jax/experimental/mosaic/gpu/examples:matmul",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_test(
    name = "matmul",
    srcs = ["//third_party/py/jax/experimental/mosaic/gpu/examples:matmul.py"],
    disable_backends = DISABLED_BACKENDS,
    disable_configs = DISABLED_CONFIGS,
    main = "//third_party/py/jax/experimental/mosaic/gpu/examples:matmul.py",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax:mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_test(
    name = "flash_attention",
    srcs = ["//third_party/py/jax/experimental/mosaic/gpu/examples:flash_attention.py"],
    disable_backends = DISABLED_BACKENDS,
    disable_configs = DISABLED_CONFIGS,
    main = "//third_party/py/jax/experimental/mosaic/gpu/examples:flash_attention.py",
    deps = [
        "//jax:mosaic_gpu",
    ] + py_deps("numpy"),
)
