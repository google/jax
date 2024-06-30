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
    name = "pallas_test",
    srcs = [
        "pallas_test.py",
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
    shard_count = 4,
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",
        "//jax:pallas_gpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_test(
    name = "gpu_attention_test",
    srcs = [
        "gpu_attention_test.py",
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
    name = "ops_test",
    srcs = [
        "ops_test.py",
    ],
    config_tags_overrides = {
        "gpu_a100_x32": {
            "ondemand": False,  # Include in presubmit.
        },
    },
    disable_backends = [
        "cpu",
    ],
    disable_configs = [
        "gpu",
        "gpu_x32",
        "gpu_a100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_h100",
    ],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_test(
    name = "indexing_test",
    srcs = [
        "indexing_test.py",
    ],
    disable_backends = [
        "gpu",
        "tpu",
    ],
    deps = [
        "//jax:pallas",
    ] + py_deps("absl/testing") + py_deps("hypothesis") + py_deps("numpy"),
)

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

jax_test(
    name = "pallas_call_tpu_test",
    srcs = ["pallas_call_tpu_test.py"],
    disable_backends = [
        "gpu",
    ],
    main = "pallas_call_tpu_test.py",
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ],
)

jax_test(
    name = "pallas_pipeline_tpu_test",
    srcs = ["pallas_pipeline_tpu_test.py"],
    disable_backends = [
        "gpu",
    ],
    main = "pallas_pipeline_tpu_test.py",
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
    name = "mosaic_gpu_test",
    srcs = [
        "mosaic_gpu_test.py",
    ],
    config_tags_overrides = {
        # TODO(slebedev): Switch to False once Mosaic GPU is unconditionally enabled.
        "gpu_h100_x32": {
            "ondemand": True,  # Include in presubmit.
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
        "gpu_a100_x32",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_h100",
    ],
    enable_configs = [
        "gpu_h100_x32",
    ],
    env = {
        "JAX_PALLAS_USE_MOSAIC_GPU": "1",
    },
    tags = ["notap"],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_test(
    name = "export_back_compat_pallas_test",
    srcs = ["export_back_compat_pallas_test.py"],
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
        "gpu_pjrt_c_api",
    ],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
    ],
    tags = [],
    deps = [
        "//jax:internal_export_back_compat_test_data",
        "//jax:internal_export_back_compat_test_util",
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
    ],
)

jax_test(
    name = "export_pallas_test",
    srcs = ["export_pallas_test.py"],
    config_tags_overrides = {
        "gpu_a100_x32": {
            "ondemand": False,  # Include in presubmit.
        },
    },
    disable_configs = [
        "gpu",
        "gpu_x32",
        "gpu_a100",
        "gpu_h100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_pjrt_c_api",
    ],
    enable_configs = [
        "gpu_a100_x32",
    ],
    tags = [],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",  # build_cleaner: keep
    ],
)

jax_test(
    name = "pallas_shape_poly_test",
    srcs = ["pallas_shape_poly_test.py"],
    config_tags_overrides = {
        "gpu_a100_x32": {
            "ondemand": False,  # Include in presubmit.
        },
    },
    disable_configs = [
        "gpu_x32",
        "gpu_h100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_pjrt_c_api",
    ],
    enable_configs = [
        "gpu_a100_x32",
    ],
    tags = [],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",  # build_cleaner: keep
        "//jax/experimental/export",
    ],
)
