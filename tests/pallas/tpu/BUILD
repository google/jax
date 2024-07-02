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

jax_test(
    name = "all_gather_test",
    srcs = [
        "all_gather_test.py",
    ],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy") + py_deps("hypothesis"),
)

jax_test(
    name = "gmm_test",
    srcs = [
        "gmm_test.py",
    ],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    shard_count = 50,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "absl/flags",
        "numpy",
        "hypothesis",
    ]),
)

jax_test(
    name = "pallas_call_test",
    srcs = ["pallas_call_test.py"],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ],
)

jax_test(
    name = "pallas_pipeline_test",
    srcs = ["pallas_pipeline_test.py"],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    shard_count = 2,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps("hypothesis"),
)

jax_test(
    name = "pallas_random_test",
    srcs = [
        "pallas_random_test.py",
    ],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_tpu",
        "//jax/_src/pallas/mosaic:random",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
    ] + py_deps("numpy"),
)

jax_test(
    name = "paged_attention_kernel_test",
    srcs = ["paged_attention_kernel_test.py"],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    shard_count = 5,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_test(
    name = "splash_attention_kernel_test",
    srcs = [
        "splash_attention_kernel_test.py",
    ],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    shard_count = 18,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy") + py_deps("hypothesis"),
)

jax_test(
    name = "splash_attention_mask_test",
    srcs = [
        "splash_attention_mask_test.py",
    ],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy") + py_deps("hypothesis"),
)
