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
    name = "attention_test",
    srcs = [
        "attention_test.py",
    ],
    config_tags_overrides = {
        "gpu_a100_x32": {
            "ondemand": False,  # Include in presubmit.
        },
    },
    disable_backends = [
        "cpu",
        "tpu",
    ],
    disable_configs = [
        "gpu",
        "gpu_x32",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_a100",
        "gpu_h100",
    ],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
    ],
    shard_count = 1,
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_gpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_test(
    name = "gpu_ops_test",
    srcs = [
        "gpu_ops_test.py",
    ],
    config_tags_overrides = {
        "gpu_a100_x32": {
            "ondemand": False,  # Include in presubmit.
        },
    },
    disable_backends = [
        "cpu",
        "tpu",
    ],
    disable_configs = [
        "gpu",
        "gpu_x32",
        "gpu_a100",
        "gpu_h100",
        "gpu_p100",
        "gpu_p100_x32",
    ],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
    ],
    shard_count = 2,
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",
        "//jax:pallas_gpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)
