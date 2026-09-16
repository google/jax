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
    "if_oss",
    "jax_generate_backend_suites",
    "jax_multiplatform_test",
    "jax_py_test",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

test_suite(
    name = "mosaic_gpu_tests",
    tags = ["mosaic_gpu_test"],
)

test_suite(
    name = "sparsecore_tests",
    tags = ["sparsecore_test"],
)

jax_multiplatform_test(
    name = "pallas_lowering_determinism_test",
    srcs = ["pallas_lowering_determinism_test.py"],
    enable_backends = [],
    enable_configs = ["tpu_grm"],
    deps = [
        "//jax/experimental:pallas_tpu",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "pallas_test",
    srcs = [
        "pallas_test.py",
    ],
    enable_backends = [
        "cpu",
        "tpu",
    ],
    enable_configs = [
        "gpu_a100",
        "gpu_h100",
        "gpu_b200",
    ],
    shard_count = {
        "cpu": 8,
        "gpu": 4,
        "tpu": 4,
    },
    deps = [
        "//jax/_src/pallas:pallas_test_util",
        "//jax/_src/pallas/mosaic_gpu:pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/experimental:pallas_triton",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "pallas_cost_estimate_test",
    srcs = [
        "pallas_cost_estimate_test.py",
    ],
    args = ["--jax_test_dut=cpu"],
    deps = [
        "//jax/_src:test_util",
        "//jax/experimental:pallas",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "utils_test",
    srcs = [
        "utils_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "cpu",
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/_src/pallas",
        "//jax/experimental:pallas",
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "ops_test",
    srcs = [
        "ops_test.py",
    ],
    disable_configs = [
        "gpu_v100",
        "gpu_v100_x32",
        "gpu_p100",
        "gpu_p100_x32",
    ],
    enable_configs = [
        "gpu_a100",
        "gpu_a100_x32",
        "gpu_h100",
        "gpu_h100_x32",
        "gpu_b200",
        "tpu_v6e",
    ],
    shard_count = {
        "cpu": 16,
        "gpu": 32,
        "tpu": 16,
    },
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/_src/pallas:pallas_test_util",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/experimental:pallas_triton",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing:flagsaver",
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_py_test(
    name = "einshape_parse_test",
    srcs = ["einshape_parse_test.py"],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/_src:test_util",
        "//jax/experimental:pallas",
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "einshape_test",
    srcs = ["einshape_test.py"],
    enable_backends = [
        "cpu",
        "tpu",
    ],
    shard_count = {
        "cpu": 2,
        "tpu": 2,
    },
    tags = ["requires-net:external"],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "ops_test_mgpu",
    srcs = [
        "ops_test.py",
    ],
    args = [
        "--jax_pallas_use_mosaic_gpu=true",
    ],
    disable_configs = [
        "gpu_v100",
        "gpu_v100_x32",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_a100",
        "gpu_a100_x32",
    ],
    enable_backends = [
        "gpu",
    ],
    enable_configs = [
        "gpu_h100",
        "gpu_h100_x32",
        "gpu_b200",
    ],
    shard_count = 16,
    tags = [
        "mosaic_gpu_test",
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/_src/pallas:pallas_test_util",
        "//jax/_src/pallas/mosaic_gpu:pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",  # build_cleaner: keep
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_triton",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing:flagsaver",
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "indexing_test",
    srcs = [
        "indexing_test.py",
    ],
    enable_backends = [
        "cpu",
        "tpu",
    ],
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "pallas_vmap_test",
    srcs = [
        "pallas_vmap_test.py",
    ],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_b200",
    ],
    shard_count = 4,
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_gpu_ops",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/experimental:pallas_triton",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mosaic_gpu_test",
    srcs = [
        "mosaic_gpu_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_a100",
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_h100",
        "gpu_b200",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    shard_count = 24,
    tags = [
        "mosaic_gpu_test",
        "noasan",
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/_src/pallas/mosaic_gpu:pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_layout_rules_test",
    srcs = [
        "mgpu_layout_rules_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "cpu",
        "cpu_x32",
        "gpu_a100",
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_h100",
        "gpu_b200",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = [
        "mosaic_gpu_test",
    ],
    deps = [
        "//jax/_src/pallas/mosaic_gpu:pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "export_back_compat_pallas_test",
    srcs = ["export_back_compat_pallas_test.py"],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_b200",
    ],
    tags = [],
    deps = [
        "//jax/_src:internal_export_back_compat_test_data",
        "//jax/_src:internal_export_back_compat_test_util",
        "//jax/_src/pallas/mosaic_gpu:pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",  # build_cleaner: keep
        "//jax/experimental:pallas_tpu_ops",  # build_cleaner: keep
        "//jax/experimental:pallas_triton",  # build_cleaner: keep
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "pallas_shape_poly_test",
    srcs = ["pallas_shape_poly_test.py"],
    disable_configs = [
        "gpu_h100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_v100_x32",
    ],
    enable_configs = [
        "gpu_a100_x32",
    ],
    tags = [],
    deps = [
        "//jax/_src/pallas:pallas_test_util",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",  # build_cleaner: keep
        "//jax/experimental:pallas_tpu",  # build_cleaner: keep
        "//jax/experimental:pallas_triton",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "pallas_error_handling_test",
    srcs = [
        "pallas_error_handling_test.py",
    ],
    enable_backends = ["tpu"],
    deps = [
        "//jax/_src/pallas/mosaic:random",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_all_gather_test",
    srcs = [
        "tpu_all_gather_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "tpu_v5e_x8",
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/experimental:mesh_utils",
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "tpu_gmm_test",
    srcs = [
        "tpu_gmm_test.py",
    ],
    enable_backends = ["tpu"],
    shard_count = 2,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notap",  # TODO(b/455849275): Timing out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "absl/flags",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_test",
    srcs = ["tpu_pallas_test.py"],
    backend_variant_args = {
        "tpu": if_oss(
            [],
            google_value = [
                "--xla_tpu_enable_mlir_verifier=ALL",
            ],
        ),
    },
    enable_backends = [
        "tpu",
        "cpu",
    ],
    enable_configs = [
        "tpu_v5e",
        "tpu_v5p",
    ],
    shard_count = 8,
    tags = [
        "noasan",
    ],
    deps = [
        "//jax/_src/pallas:pallas_test_util",
        "//jax/experimental:mesh_utils",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/extend",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_call_print_test",
    srcs = ["tpu_pallas_call_print_test.py"],
    # The flag is necessary for ``pl.debug_print`` tests to work on TPU.
    args = ["--logtostderr"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e",
        "tpu_v5p",
    ],
    env = if_oss(
        {
            "TPU_STDERR_LOG_LEVEL": "0",
        },
        google_value = {},
    ),
    shard_count = 8,
    deps = [
        "//jax/_src/pallas:pallas_test_util",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/extend",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_pallas_distributed_multiprocess_test",
    srcs = ["gpu_pallas_distributed_test.py"],
    args = [
        "--num_processes=2",
        "--gpus_per_process=1",
    ],
    enable_backends = [],
    enable_configs = [
        # Need to use 2 physical GPUs since test checks P2P memory access unavailable with MIG
        # partitioning.
        "gpu_h100x2_full",
        "gpu_b200x2",
    ],
    env = {
        "MULTIPROCESS_TEST": "1",
    },
    tags = [
        "mosaic_gpu_test",
        "multiaccelerator",
    ],
    deps = [
        "//jax/_src:test_multiprocess",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/extend",
    ] + py_deps([
        "portpicker",
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_pallas_distributed_test",
    srcs = ["gpu_pallas_distributed_test.py"],
    config_tags_overrides = {
        "gpu_b200x2": {"ondemand": False},
    },
    enable_backends = [],
    enable_configs = [
        # Need to use 2 physical GPUs since test checks P2P memory access unavailable with MIG
        # partitioning.
        "gpu_h100x2_full",
        "gpu_b200x2",
        # "gpu_gb300x2",  # TODO(b/548528820): Re-enable this.
    ],
    tags = [
        "mosaic_gpu_test",
        "multiaccelerator",
    ],
    deps = [
        "//jax/_src:test_multiprocess",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/extend",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_ops_test",
    srcs = [
        "tpu_ops_test.py",
    ],
    enable_backends = ["tpu"],
    shard_count = 8,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/_src/pallas:pallas_test_util",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/experimental:pallas_triton",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_distributed_test",
    srcs = ["tpu_pallas_distributed_test.py"],
    # TODO(sharadmv): fix timeouts
    disable_configs = [
        "tpu_7x",
        "tpu_7x_x4",
    ],
    enable_backends = ["tpu"],
    tags = ["multiaccelerator"],
    deps = [
        "//jax/experimental:mesh_utils",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/extend",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_pipeline_test",
    srcs = ["tpu_pallas_pipeline_test.py"],
    args = if_oss(
        [],
        google_value = [
            # Timeout on deadlocks for a better error message.
            "--xla_tpu_debug_sflag_wait_timeout_ms=500",
            "--xla_tpu_debug_sflag_wait_shalt_on_detection",
        ],
    ),
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e_x8",
        "tpu_v5p",
    ],
    minimal_shard_count = 2,
    shard_count = 5,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/experimental:hijax",
        "//jax/experimental:mesh_utils",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/extend",
    ] + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_async_test",
    srcs = ["tpu_pallas_async_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e_x8",
        "tpu_v5p",
        "tpu_v5p_x4",
    ],
    deps = [
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_memory_space_test",
    srcs = ["tpu_pallas_memory_space_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5p",
    ],
    deps = [
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_side_effects_test",
    srcs = ["tpu_side_effects_test.py"],
    enable_backends = ["tpu"],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_state_test",
    srcs = ["tpu_pallas_state_test.py"],
    enable_backends = ["tpu"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/experimental:pallas_tpu",
        "//jax/extend",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_random_test",
    srcs = [
        "tpu_pallas_random_test.py",
    ],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5p_x4",
    ],
    deps = [
        "//jax/_src/pallas/mosaic:random",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_interpret_test",
    srcs = [
        "tpu_pallas_interpret_test.py",
    ],
    enable_backends = ["cpu"],
    tags = [
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_interpret_thread_map_test",
    srcs = [
        "tpu_pallas_interpret_thread_map_test.py",
    ],
    enable_backends = ["cpu"],
    deps = [
        "//jax/experimental",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_interpret_distributed_test",
    srcs = [
        "tpu_pallas_interpret_distributed_test.py",
    ],
    enable_backends = ["cpu"],
    tags = [
        "noasan",  # Times out.
        "nomsan",
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_paged_attention_kernel_test",
    srcs = ["tpu_paged_attention_kernel_test.py"],
    disable_configs = [
        "tpu_v5p",
    ],
    enable_backends = ["tpu"],
    minimal_shard_count = 10,
    shard_count = 10,
    tags = [
        "noasan",  # Times out.
        "noassert",
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_ragged_paged_attention_test",
    srcs = ["tpu_ragged_paged_attention_test.py"],
    disable_configs = [
        "tpu_v5p",
    ],
    enable_backends = ["tpu"],
    minimal_shard_count = 8,
    shard_count = 24,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_splash_attention_kernel_test",
    srcs = [
        "tpu_splash_attention_kernel_test.py",
    ],
    disable_configs = [
        "tpu_7x_x4",  # not a multi-chip test
    ],
    enable_backends = ["tpu"],
    minimal_shard_count = 8,
    shard_count = 24,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/_src/pallas:pallas_test_util",
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "tpu_splash_attention_kernel_sharded_test",
    srcs = ["tpu_splash_attention_kernel_sharded_test.py"],
    enable_configs = [
        "tpu_v5e_x8",
        "tpu_v5p_x4",
    ],
    shard_count = 10,
    deps = [
        "//jax/_src/pallas:pallas_test_util",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
        "//jax/extend",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

# This test doesn't need a TPU; it only tests numpy-using helpers.
jax_py_test(
    name = "tpu_splash_attention_mask_test",
    srcs = [
        "tpu_splash_attention_mask_test.py",
    ],
    deps = [
        "//jax",
        "//jax/_src:test_util",
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "gpu_attention_test",
    srcs = [
        "gpu_attention_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_b200",
    ],
    shard_count = 1,
    tags = [
        "noasan",  # Times out.
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_gpu_ops",
        "//jax/experimental:pallas_triton",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_ops_test",
    srcs = [
        "gpu_ops_test.py",
    ],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_b200",
    ],
    shard_count = 20,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_gpu_ops",
        "//jax/experimental:pallas_triton",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_paged_attention_test",
    srcs = [
        "gpu_paged_attention_test.py",
    ],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_b200",
    ],
    shard_count = 12,
    tags = [
        "noasan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_gpu_ops",
        "//jax/experimental:pallas_triton",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "triton_gpu_info_test",
    srcs = [
        "triton_gpu_info_test.py",
    ],
    enable_backends = ["gpu"],
    shard_count = 1,
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_triton",
    ] + py_deps([
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "triton_pallas_test",
    srcs = [
        "triton_pallas_test.py",
    ],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_h100_x32",
        "gpu_b200",
    ],
    shard_count = 1,
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_triton",
    ] + py_deps([
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_attention_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:attention_mgpu.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_full",
        "gpu_b200_full",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_attention_test",
    srcs = ["mgpu_attention_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_x32",
        "gpu_h100",
        "gpu_b200",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    shard_count = 8,
    tags = [
        "mosaic_gpu_test",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_examples_test",
    srcs = ["mgpu_examples_test.py"],
    enable_backends = [],
    enable_configs = ["gpu_b200"],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    shard_count = 8,
    tags = [
        "mosaic_gpu_test",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_matmul_test",
    srcs = ["mgpu_matmul_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_b200",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    shard_count = 8,
    tags = ["mosaic_gpu_test"],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "hopper_matmul_mgpu_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:hopper_matmul_mgpu.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100_full"],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "hopper_mixed_type_matmul_mgpu_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:hopper_mixed_type_matmul_mgpu.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100_full"],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "blackwell_matmul_mgpu_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:blackwell_matmul_mgpu.py"],
    enable_backends = [],
    enable_configs = ["gpu_b200_full"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "blackwell_ragged_dot_mgpu_run",
    srcs = [
        "//jax/experimental/pallas/ops/gpu:blackwell_matmul_mgpu.py",
        "//jax/experimental/pallas/ops/gpu:blackwell_ragged_dot_mgpu.py",
        "//jax/experimental/pallas/ops/gpu:ragged_dot_mgpu.py",
    ],
    enable_backends = [],
    enable_configs = ["gpu_b200_full"],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    main = "//jax/experimental/pallas/ops/gpu:blackwell_ragged_dot_mgpu.py",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_ragged_dot_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:ragged_dot_mgpu.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_full",
        "gpu_b200_full",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "mgpu_transposed_ragged_dot_run",
    srcs = [
        "//jax/experimental/pallas/ops/gpu:transposed_ragged_dot_mgpu.py",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_full",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "mgpu_ragged_dot_test",
    size = "large",  # Increased timout to account for the extra tests.
    srcs = ["mgpu_ragged_dot_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_b200",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    shard_count = 20,
    tags = [
        "mosaic_gpu_test",
        "noasan",  # Times out.
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_collective_matmul_test",
    srcs = ["mgpu_collective_matmul_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100x2_full",  # We need a 2 physical GPUs since otherwise the test will be skipped
        "gpu_b200x2",
    ],
    tags = [
        "manual",
        "multiaccelerator",
        "notap",
    ],
    deps = [
        "//jax/_src:test_multiprocess",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "mgpu_torch_test",
    srcs = ["mgpu_torch_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_x32",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = [
        "noasan",
        "notap",  # TODO(b/477964663): Broken by PyTorch 2.10.0 upgrade.
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy") + py_deps("torch"),
)

jax_multiplatform_test(
    name = "mgpu_collective_matmul_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:collective_matmul_mgpu.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100x2_full",
    ],
    tags = [
        "manual",
        "multiaccelerator",
        "notap",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
    ],
)

jax_multiplatform_test(
    name = "mgpu_reduce_scatter_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:reduce_scatter_mgpu.py"],
    args = [
        "--num_processes=2",
        "--gpus_per_process=1",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_h100x2_full",
    ],
    tags = [
        "manual",
        "multiaccelerator",
        "notap",
    ],
    deps = [
        "//jax/_src:test_multiprocess",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
    ],
)

jax_multiplatform_test(
    name = "mgpu_all_gather_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:all_gather_mgpu.py"],
    args = [
        "--num_processes=2",
        "--gpus_per_process=1",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_h100x2_full",
    ],
    tags = [
        "manual",
        "mosaic_gpu_test",
        "multiaccelerator",
        "notap",
    ],
    deps = [
        "//jax/_src:test_multiprocess",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/extend",
    ] + py_deps([
        "portpicker",
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "fuser_block_spec_test",
    srcs = [
        "fuser_block_spec_test.py",
    ],
    disable_configs = [
        "cpu",
    ],
    enable_backends = ["cpu"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/_src/pallas/fuser",
        "//jax/experimental:pallas",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "fuser_utils_test",
    srcs = [
        "fuser_utils_test.py",
    ],
    disable_configs = [
        "cpu",
    ],
    enable_backends = ["cpu"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/_src/pallas/fuser",
        "//jax/experimental:pallas",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "fusion_test",
    srcs = [
        "fusion_test.py",
    ],
    disable_configs = [
        "cpu",
    ],
    enable_backends = ["cpu"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_fuser",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_fusible_matmul_test",
    srcs = ["tpu_fusible_matmul_test.py"],
    disable_configs = [
        "tpu_v3",
    ],
    enable_backends = ["tpu"],
    shard_count = 4,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/experimental:pallas_fuser",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_fusible_matmul_with_stateful_fusions_test",
    srcs = ["tpu_fusible_matmul_with_stateful_fusions_test.py"],
    disable_configs = [
        "tpu_v3",
    ],
    enable_backends = ["tpu"],
    # shard_count = 4,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/experimental:pallas_fuser",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_sparsecore_pallas_test",
    srcs = ["tpu_sparsecore_pallas_test.py"],
    backend_variant_args = {
        "tpu": if_oss(
            [],
            google_value = [
                "--xla_tpu_enable_mlir_verifier=ALL",
                "--xla_sc_assert_level=san",
            ],
        ),
        "tpu_8i_grm": [
            "--xla_tpu_enable_mlir_verifier=ALL",
            "--xla_sc_assert_level=san",
        ],
    },
    enable_backends = [],
    enable_configs = [
        "tpu_v5p",
        "tpu_v6e",
        "tpu_7x",
        "tpu_8i_grm",
        "cpu",
    ],
    shard_count = 10,
    tags = [
        "cpu:4",
        "sparsecore_test",
    ],
    deps = [
        "//jax/_src:hypothesis_test_util",
        "//jax/_src:mesh",
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:tpu_info",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_sc",
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_mpmd_test",
    srcs = ["tpu_pallas_mpmd_test.py"],
    args = if_oss(
        [],
        google_value = [
            "--xla_tpu_enable_mlir_verifier=ALL",
            "--xla_sc_assert_level=san",
            "--xla_mosaic_unsafe_allow_multicore_remote_dma=true",
        ],
    ),
    enable_backends = [],
    enable_configs = [
        "tpu_v6e",
        "tpu_7x",
    ],
    tags = ["sparsecore_test"],
    deps = [
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:primitives",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_fuser",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_sc",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_sparsecore_pallas_debug_check_test",
    srcs = ["tpu_sparsecore_pallas_debug_check_test.py"],
    backend_variant_args = {
        "tpu": if_oss(
            [],
            google_value = [
                "--xla_tpu_enable_mlir_verifier=ALL",
                "--xla_sc_assert_level=san",
            ],
        ),
    },
    enable_backends = [],
    enable_configs = [
        "tpu_v5p",
        "tpu_v6e",
        "tpu_7x",
        "cpu",  # Just to make sure we at least check the shard_count in presubmit.
    ],
    shard_count = 4,
    tags = ["sparsecore_test"],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_sc",
    ] + py_deps([
        "numpy",
        "absl/testing",
        "absl/flags",
    ]),
)

jax_multiplatform_test(
    name = "tpu_sparsecore_pallas_distributed_test",
    srcs = ["tpu_sparsecore_pallas_distributed_test.py"],
    args = if_oss(
        [],
        google_value = [
            "--xla_tpu_enable_mlir_verifier=ALL",
            "--xla_sc_assert_level=san",
        ],
    ),
    config_tags_overrides = {
        # TODO(slebedev): Flip to False once we have more v5p capacity.
        "tpu_v5p_x4": {"ondemand": True},
        "tpu_v6e_x8": {"ondemand": True},
    },
    enable_backends = [],
    enable_configs = [
        "tpu_v5p_x4",
        "tpu_v6e_x8",
        "tpu_7x_x4",
    ],
    tags = ["sparsecore_test"],
    deps = [
        "//jax/experimental:mesh_utils",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_sc",
    ] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "tpu_info_test",
    srcs = ["tpu_info_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_7x",
    ],
    deps = [
        "//jax/experimental:pallas_tpu",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "gpu_pallas_interpret_test",
    size = "large",
    srcs = ["gpu_pallas_interpret_test.py"],
    enable_backends = ["cpu"],
    shard_count = 12,
    deps = [
        "//jax/_src/pallas/mosaic_gpu/interpret:interpret_pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_pallas_interpret_race_detection_test",
    srcs = ["gpu_pallas_interpret_race_detection_test.py"],
    enable_backends = ["cpu"],
    shard_count = 4,
    deps = [
        "//jax/_src/pallas/mosaic_gpu/interpret:interpret_pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_pallas_interpret_distributed_test",
    srcs = ["gpu_pallas_interpret_distributed_test.py"],
    enable_backends = ["cpu"],
    shard_count = 8,
    tags = [
        # Times out.
        "noasan",
        "notsan",
        "nomsan",
    ],
    deps = [
        "//jax/_src/pallas/mosaic_gpu/interpret:interpret_pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/experimental:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_pallas_interpret_cluster_barrier_test",
    srcs = ["gpu_pallas_interpret_cluster_barrier_test.py"],
    enable_backends = ["cpu"],
    shard_count = 8,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax/_src/pallas/mosaic_gpu/interpret:interpret_pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_pallas_interpret_tcgen05_race_detection_test",
    srcs = ["gpu_pallas_interpret_tcgen05_race_detection_test.py"],
    enable_backends = ["cpu"],
    shard_count = 6,
    deps = [
        "//jax/_src/pallas/mosaic_gpu/interpret:interpret_pallas_call",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_trace_value_test",
    srcs = ["tpu_trace_value_test.py"],
    args = if_oss(
        [],
        google_value = [
            "--xla_tpu_enable_mlir_verifier=ALL",
            "--xla_sc_assert_level=san",
        ],
    ),
    enable_backends = ["tpu"],
    enable_configs = ["tpu_7x"],
    deps = [
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
        "//jax/experimental:pallas_tpu_sc",
    ] + py_deps([
        "absl/testing",
    ]),
)
