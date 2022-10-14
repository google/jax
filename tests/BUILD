# Copyright 2018 The JAX Authors.
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
    "jax_test_file_visibility",
    "py_deps",
    "pytype_library",
    "pytype_test",
)

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

jax_generate_backend_suites()

jax_test(
    name = "api_test",
    srcs = ["api_test.py"],
    pjrt_c_api_bypass = True,
    shard_count = 10,
)

jax_test(
    name = "dynamic_api_test",
    srcs = ["dynamic_api_test.py"],
    shard_count = 2,
)

jax_test(
    name = "api_util_test",
    srcs = ["api_util_test.py"],
)

jax_test(
    name = "array_interoperability_test",
    srcs = ["array_interoperability_test.py"],
    disable_backends = ["tpu"],
    deps = py_deps("tensorflow"),
)

jax_test(
    name = "batching_test",
    srcs = ["batching_test.py"],
    shard_count = {
        "gpu": 5,
    },
)

jax_test(
    name = "callback_test",
    srcs = ["callback_test.py"],
    deps = ["//jax:callback"],
)

jax_test(
    name = "core_test",
    srcs = ["core_test.py"],
    shard_count = {
        "cpu": 5,
        "gpu": 10,
    },
)

jax_test(
    name = "custom_object_test",
    srcs = ["custom_object_test.py"],
)

jax_test(
    name = "debug_nans_test",
    srcs = ["debug_nans_test.py"],
)

py_test(
    name = "multiprocess_gpu_test",
    srcs = ["multiprocess_gpu_test.py"],
    args = [
        "--exclude_test_targets=MultiProcessGpuTest",
    ],
    tags = ["manual"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("portpicker"),
)

jax_test(
    name = "dtypes_test",
    srcs = ["dtypes_test.py"],
)

py_test(
    name = "errors_test",
    srcs = ["errors_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ],
)

jax_test(
    name = "fft_test",
    srcs = ["fft_test.py"],
    backend_tags = {
        "tpu": ["notsan"],  # Times out on TPU with tsan.
    },
    shard_count = {
        "tpu": 20,
    },
)

jax_test(
    name = "generated_fun_test",
    srcs = ["generated_fun_test.py"],
)

jax_test(
    name = "lobpcg_test",
    srcs = ["lobpcg_test.py"],
    env = {"LOBPCG_EMIT_DEBUG_PLOTS": "1"},
    shard_count = {
        "cpu": 48,
        "gpu": 48,
        "tpu": 48,
    },
    deps = [
        "//jax:experimental_sparse",
    ] + py_deps("matplotlib"),
)

jax_test(
    name = "svd_test",
    srcs = ["svd_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 10,
    },
)

py_test(
    name = "xla_interpreter_test",
    srcs = ["xla_interpreter_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ],
)

jax_test(
    name = "xmap_test",
    srcs = ["xmap_test.py"],
    pjrt_c_api_bypass = True,
    shard_count = {
        "cpu": 10,
        "gpu": 4,
        "tpu": 4,
    },
    tags = ["multiaccelerator"],
    deps = [
        "//jax:maps",
    ],
)

jax_test(
    name = "pjit_test",
    srcs = ["pjit_test.py"],
    backend_tags = {
        "tpu": ["notsan"],  # Times out under tsan.
    },
    pjrt_c_api_bypass = True,
    shard_count = {
        "cpu": 5,
        "gpu": 5,
        "tpu": 5,
    },
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
    ],
)

jax_test(
    name = "global_device_array_test",
    srcs = ["global_device_array_test.py"],
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
    ],
)

jax_test(
    name = "array_test",
    srcs = ["array_test.py"],
    pjrt_c_api_bypass = True,
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
    ],
)

jax_test(
    name = "remote_transfer_test",
    srcs = ["remote_transfer_test.py"],
    disable_backends = [
        "gpu",
        "cpu",
    ],
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
    ],
)

jax_test(
    name = "image_test",
    srcs = ["image_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 20,
        "tpu": 10,
        "iree": 10,
    },
    deps = py_deps("pil") + py_deps("tensorflow"),
)

jax_test(
    name = "infeed_test",
    srcs = ["infeed_test.py"],
    pjrt_c_api_bypass = True,
    deps = [
        "//jax:experimental_host_callback",
    ],
)

jax_test(
    name = "jax_jit_test",
    srcs = ["jax_jit_test.py"],
    main = "jax_jit_test.py",
)

py_test(
    name = "jax_to_ir_test",
    srcs = ["jax_to_ir_test.py"],
    deps = [
        "//jax:test_util",
        "//jax/experimental/jax2tf",
        "//jax/tools:jax_to_ir",
    ] + py_deps("tensorflow"),
)

jax_test(
    name = "jaxpr_util_test",
    srcs = ["jaxpr_util_test.py"],
)

jax_test(
    name = "jet_test",
    srcs = ["jet_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
    },
    deps = [
        "//jax:jet",
        "//jax:stax",
    ],
)

jax_test(
    name = "lax_control_flow_test",
    srcs = ["lax_control_flow_test.py"],
    shard_count = {
        "cpu": 30,
        "gpu": 40,
        "tpu": 30,
        "iree": 10,
    },
)

jax_test(
    name = "custom_root_test",
    srcs = ["custom_root_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 10,
        "iree": 10,
    },
)

jax_test(
    name = "custom_linear_solve_test",
    srcs = ["custom_linear_solve_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 10,
        "iree": 10,
    },
)

jax_test(
    name = "lax_numpy_test",
    srcs = ["lax_numpy_test.py"],
    pjrt_c_api_bypass = True,
    shard_count = {
        "cpu": 20,
        "gpu": 20,
        "tpu": 10,
    },
)

jax_test(
    name = "lax_numpy_operators_test",
    srcs = ["lax_numpy_operators_test.py"],
    pjrt_c_api_bypass = True,
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 5,
    },
)

jax_test(
    name = "lax_numpy_reducers_test",
    srcs = ["lax_numpy_reducers_test.py"],
    pjrt_c_api_bypass = True,
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 5,
    },
)

jax_test(
    name = "lax_numpy_indexing_test",
    srcs = ["lax_numpy_indexing_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 10,
        "iree": 10,
    },
)

jax_test(
    name = "lax_numpy_einsum_test",
    srcs = ["lax_numpy_einsum_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 10,
        "iree": 10,
    },
)

jax_test(
    name = "lax_numpy_vectorize_test",
    srcs = ["lax_numpy_vectorize_test.py"],
)

jax_test(
    name = "lax_scipy_test",
    srcs = ["lax_scipy_test.py"],
    backend_tags = {
        "tpu": ["noasan"],  # Test times out.
    },
    shard_count = {
        "cpu": 20,
        "gpu": 40,
        "tpu": 10,
        "iree": 10,
    },
)

jax_test(
    name = "lax_scipy_sparse_test",
    srcs = ["lax_scipy_sparse_test.py"],
    backend_tags = {
        "cpu": ["nomsan"],  # Test fails under msan because of fortran code inside scipy.
    },
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 10,
        "iree": 10,
    },
)

jax_test(
    name = "lax_test",
    srcs = ["lax_test.py"],
    pjrt_c_api_bypass = True,
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 20,
        "iree": 40,
    },
)

pytype_library(
    name = "lax_test_lib",
    srcs = ["lax_test.py"],
    srcs_version = "PY3",
    deps = ["//jax"],
)

pytype_library(
    name = "lax_vmap_test_lib",
    testonly = 1,
    srcs = ["lax_vmap_test.py"],
    srcs_version = "PY3",
    deps = [
        ":lax_test_lib",
        "//jax",
        "//jax:test_util",
    ],
)

jax_test(
    name = "lax_autodiff_test",
    srcs = ["lax_autodiff_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 20,
        "iree": 40,
    },
    deps = [":lax_test_lib"],
)

jax_test(
    name = "lax_vmap_test",
    srcs = ["lax_vmap_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 20,
        "iree": 40,
    },
    deps = [":lax_test_lib"],
)

py_test(
    name = "lazy_loader_test",
    srcs = [
        "lazy_loader_module/__init__.py",
        "lazy_loader_module/lazy_test_submodule.py",
        "lazy_loader_test.py",
    ],
    deps = [
        "//jax:test_util",
    ],
)

jax_test(
    name = "linalg_test",
    srcs = ["linalg_test.py"],
    backend_tags = {
        "tpu": [
            "cpu:8",
            "noasan",  # Times out.
            "nodebug",  # Times out.
            "notsan",  # Times out.
        ],
    },
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 10,
        "iree": 20,
    },
)

jax_test(
    name = "metadata_test",
    srcs = ["metadata_test.py"],
    disable_backends = [
        "gpu",
        "tpu",
    ],
)

jax_test(
    name = "multibackend_test",
    srcs = ["multibackend_test.py"],
)

jax_test(
    name = "multi_device_test",
    srcs = ["multi_device_test.py"],
    disable_backends = [
        "gpu",
        "tpu",
    ],
)

jax_test(
    name = "nn_test",
    srcs = ["nn_test.py"],
)

jax_test(
    name = "optimizers_test",
    srcs = ["optimizers_test.py"],
    deps = ["//jax:optimizers"],
)

jax_test(
    name = "pickle_test",
    srcs = ["pickle_test.py"],
    deps = [
        "//jax:experimental",
    ] + py_deps("cloudpickle"),
)

jax_test(
    name = "pmap_test",
    srcs = ["pmap_test.py"],
    backend_tags = {
        "tpu": [
            "noasan",  # Times out.
            "nodebug",  # Times out.
            "notsan",  # Times out.
        ],
    },
    pjrt_c_api_bypass = True,
    shard_count = {
        "cpu": 30,
        "gpu": 30,
        "tpu": 30,
    },
    tags = ["multiaccelerator"],
    deps = [
        ":lax_test_lib",
        ":lax_vmap_test_lib",
    ],
)

jax_test(
    name = "polynomial_test",
    srcs = ["polynomial_test.py"],
    # No implementation of nonsymmetric Eigendecomposition.
    disable_backends = [
        "gpu",
        "tpu",
    ],
    shard_count = {
        "cpu": 10,
    },
    # This test ends up calling Fortran code that initializes some memory and
    # passes it to C code. MSan is not able to detect that the memory was
    # initialized by Fortran, and it makes the test fail. This can usually be
    # fixed by annotating the memory with `ANNOTATE_MEMORY_IS_INITIALIZED`, but
    # in this case there's not a good place to do it, see b/197635968#comment19
    # for details.
    tags = ["nomsan"],
)

jax_test(
    name = "heap_profiler_test",
    srcs = ["heap_profiler_test.py"],
    disable_backends = [
        "gpu",
        "tpu",
    ],
)

jax_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    disable_backends = [
        "gpu",
        "tpu",
    ],
)

jax_test(
    name = "qdwh_test",
    srcs = ["qdwh_test.py"],
)

jax_test(
    name = "random_test",
    srcs = ["random_test.py"],
    shard_count = {
        "cpu": 20,
        "gpu": 20,
        "tpu": 20,
        "iree": 20,
    },
)

# TODO(b/199564969): remove once we always enable_custom_prng
jax_test(
    name = "random_test_with_custom_prng",
    srcs = ["random_test.py"],
    args = ["--jax_enable_custom_prng=true"],
    backend_tags = {
        "cpu": ["noasan"],  # Times out under asan.
    },
    main = "random_test.py",
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 40,
        "iree": 20,
    },
)

jax_test(
    name = "scipy_fft_test",
    srcs = ["scipy_fft_test.py"],
)

jax_test(
    name = "scipy_interpolate_test",
    srcs = ["scipy_interpolate_test.py"],
)

jax_test(
    name = "scipy_ndimage_test",
    srcs = ["scipy_ndimage_test.py"],
)

jax_test(
    name = "scipy_optimize_test",
    srcs = ["scipy_optimize_test.py"],
)

jax_test(
    name = "scipy_signal_test",
    srcs = ["scipy_signal_test.py"],
    backend_tags = {
        "cpu": [
            "noasan",  # Test times out under asan.
        ],
        "tpu": [
            "noasan",
            "notsan",
        ],  # Test times out under asan/tsan.
    },
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 5,
    },
)

jax_test(
    name = "scipy_stats_test",
    srcs = ["scipy_stats_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 20,
        "tpu": 10,
        "iree": 10,
    },
)

jax_test(
    name = "sparse_test",
    srcs = ["sparse_test.py"],
    args = ["--jax_bcoo_cusparse_lowering=true"],
    backend_tags = {
        "cpu": [
            "noasan",  # Test times out under asan.
        ],
    },
    shard_count = {
        "cpu": 20,
        "gpu": 40,
        "tpu": 40,
        "iree": 10,
    },
    deps = [
        "//jax:experimental_sparse",
    ] + py_deps("scipy"),
)

jax_test(
    name = "sparsify_test",
    srcs = ["sparsify_test.py"],
    args = ["--jax_bcoo_cusparse_lowering=true"],
    shard_count = {
        "gpu": 20,
        "tpu": 10,
    },
    deps = [
        "//jax:experimental_sparse",
    ],
)

jax_test(
    name = "stack_test",
    srcs = ["stack_test.py"],
)

jax_test(
    name = "checkify_test",
    srcs = ["checkify_test.py"],
)

jax_test(
    name = "stax_test",
    srcs = ["stax_test.py"],
    shard_count = {
        "cpu": 5,
        "gpu": 5,
        "iree": 5,
    },
    deps = ["//jax:stax"],
)

jax_test(
    name = "linear_search_test",
    srcs = ["third_party/scipy/line_search_test.py"],
    main = "third_party/scipy/line_search_test.py",
)

py_test(
    name = "tree_util_test",
    srcs = ["tree_util_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ],
)

pytype_test(
    name = "typing_test",
    srcs = ["typing_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ],
)

py_test(
    name = "version_test",
    srcs = ["version_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ],
)

py_test(
    name = "xla_bridge_test",
    srcs = ["xla_bridge_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/logging"),
)

py_test(
    name = "gfile_cache_test",
    srcs = ["gfile_cache_test.py"],
    deps = [
        "//jax",
        "//jax:compilation_cache",
        "//jax:test_util",
    ],
)

jax_test(
    name = "compilation_cache_test",
    srcs = ["compilation_cache_test.py"],
    backend_tags = {
        "tpu": ["nomsan"],  # TODO(b/213388298): this test fails msan.
    },
    pjrt_c_api_bypass = True,
    deps = [
        "//jax:compilation_cache",
        "//jax:experimental",
    ],
)

jax_test(
    name = "ode_test",
    srcs = ["ode_test.py"],
    shard_count = {
        "cpu": 10,
    },
    deps = ["//jax:ode"],
)

jax_test(
    name = "host_callback_test",
    srcs = ["host_callback_test.py"],
    args = ["--jax_host_callback_outfeed=true"],
    pjrt_c_api_bypass = True,
    deps = [
        "//jax:experimental",
        "//jax:experimental_host_callback",
        "//jax:ode",
    ],
)

jax_test(
    name = "host_callback_custom_call_test",
    srcs = ["host_callback_test.py"],
    args = ["--jax_host_callback_outfeed=false"],
    disable_backends = [
        "gpu",
        "tpu",  # On TPU we always use outfeed
    ],
    main = "host_callback_test.py",
    shard_count = {
        "gpu": 5,
    },
    deps = [
        "//jax:experimental",
        "//jax:experimental_host_callback",
        "//jax:ode",
    ],
)

jax_test(
    name = "host_callback_to_tf_test",
    srcs = ["host_callback_to_tf_test.py"],
    pjrt_c_api_bypass = True,
    deps = [
        "//jax:experimental_host_callback",
        "//jax:ode",
    ] + py_deps("tensorflow"),
)

jax_test(
    name = "x64_context_test",
    srcs = ["x64_context_test.py"],
    deps = [
        "//jax:experimental",
    ],
)

jax_test(
    name = "ann_test",
    srcs = ["ann_test.py"],
    shard_count = 2,
    deps = [
        ":lax_test_lib",
    ],
)

py_test(
    name = "mesh_utils_test",
    srcs = ["mesh_utils_test.py"],
    deps = [
        "//jax",
        "//jax:mesh_utils",
        "//jax:test_util",
    ],
)

jax_test(
    name = "transfer_guard_test",
    srcs = ["transfer_guard_test.py"],
)

jax_test(
    name = "program_eviction_test",
    srcs = ["program_eviction_test.py"],
    args = [
        "--jax_tpu_runtime=tfrt",
    ],
    disable_backends = [
        "cpu",
        "gpu",
    ],
    disable_configs = [
        "tpu_pjrt_c_api",
    ],
    deps = [
        "//jax:experimental_program_eviction",
    ],
)

jax_test(
    name = "name_stack_test",
    srcs = ["name_stack_test.py"],
)

jax_test(
    name = "jaxpr_effects_test",
    srcs = ["jaxpr_effects_test.py"],
    enable_configs = [
        "gpu",
        "cpu",
    ],
)

jax_test(
    name = "debugging_primitives_test",
    srcs = ["debugging_primitives_test.py"],
    enable_configs = [
        "gpu",
        "cpu",
    ],
)

jax_test(
    name = "python_callback_test",
    srcs = ["python_callback_test.py"],
)

jax_test(
    name = "debugger_test",
    srcs = ["debugger_test.py"],
    enable_configs = [
        "gpu",
        "cpu",
    ],
)

jax_test(
    name = "state_test",
    srcs = ["state_test.py"],
    enable_configs = [
        "gpu",
        "cpu",
    ],
)

jax_test(
    name = "for_loop_test",
    srcs = ["for_loop_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 10,
    },
)

jax_test(
    name = "clear_backends_test",
    srcs = ["clear_backends_test.py"],
)

exports_files(
    [
        "api_test.py",
        "pmap_test.py",
        "compilation_cache_test.py",
        "pjit_test.py",
        "python_callback_test.py",
        "transfer_guard_test.py",
        "program_eviction_test.py",
    ],
    visibility = jax_test_file_visibility,
)

# This filegroup specifies the set of tests known to Bazel, used for a test that
# verifies every test has a Bazel test rule.
# If a test isn't meant to be tested with Bazel, add it to the exclude list.
filegroup(
    name = "all_tests",
    srcs = glob(
        include = [
            "*_test.py",
            "third_party/*/*_test.py",
        ],
        exclude = [],
    ) + ["BUILD"],
    visibility = [
        "//third_party/py/jax:__subpackages__",
    ],
)
