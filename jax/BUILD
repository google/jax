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

# JAX is Autograd and XLA

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "//jaxlib:jax.bzl",
    "if_building_jaxlib",
    "jax_extend_internal_users",
    "jax_extra_deps",
    "jax_internal_export_back_compat_test_util_visibility",
    "jax_internal_packages",
    "jax_internal_test_harnesses_visibility",
    "jax_test_util_visibility",
    "jax_visibility",
    "mosaic_internal_users",
    "pallas_gpu_internal_users",
    "pallas_tpu_internal_users",
    "py_deps",
    "py_library_providing_imports_info",
    "pytype_library",
    "pytype_strict_library",
)
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = [],
    default_visibility = [":internal"],
)

licenses(["notice"])

# If this flag is true, jaxlib should be built by bazel. If false, then we do not build jaxlib and
# assume it has been installed, e.g., by `pip`.
bool_flag(
    name = "build_jaxlib",
    build_setting_default = True,
)

config_setting(
    name = "enable_jaxlib_build",
    flag_values = {
        ":build_jaxlib": "True",
    },
)

# When `build_cuda_plugin_from_source` is true, it assumes running `bazel test` without preinstalled
# cuda plugin.
bool_flag(
    name = "build_cuda_plugin_from_source",
    build_setting_default = False,
)

config_setting(
    name = "enable_build_cuda_plugin_from_source",
    flag_values = {
        ":build_cuda_plugin_from_source": "True",
    },
)

exports_files([
    "LICENSE",
    "version.py",
])

# Packages that have access to JAX-internal implementation details.
package_group(
    name = "internal",
    packages = [
        "//...",
    ] + jax_internal_packages,
)

package_group(
    name = "jax_extend_users",
    includes = [":internal"],
    packages = [
        # Intentionally avoid jax dependencies on jax.extend.
        # See https://jax.readthedocs.io/en/latest/jep/15856-jex.html
        "//third_party/py/jax/tests/...",
    ] + jax_extend_internal_users,
)

package_group(
    name = "mosaic_users",
    packages = [
        "//...",
    ] + mosaic_internal_users,
)

package_group(
    name = "pallas_gpu_users",
    packages = [
        "//...",
    ] + pallas_gpu_internal_users,
)

package_group(
    name = "pallas_tpu_users",
    packages = [
        "//...",
    ] + pallas_tpu_internal_users,
)

# JAX-private test utilities.
py_library(
    # This build target is required in order to use private test utilities in jax._src.test_util,
    # and its visibility is intentionally restricted to discourage its use outside JAX itself.
    # JAX does provide some public test utilities (see jax/test_util.py);
    # these are available in jax.test_util via the standard :jax target.
    name = "test_util",
    testonly = 1,
    srcs = [
        "_src/test_util.py",
    ],
    visibility = [
        ":internal",
    ] + jax_test_util_visibility,
    deps = [
        ":jax",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

# TODO(necula): break the internal_test_util into smaller build targets.
py_library(
    name = "internal_test_util",
    testonly = 1,
    srcs = [
        "_src/internal_test_util/deprecation_module.py",
        "_src/internal_test_util/lax_test_util.py",
    ] + glob(
        [
            "_src/internal_test_util/lazy_loader_module/*.py",
        ],
    ),
    visibility = [":internal"],
    deps = [
        ":jax",
    ] + py_deps("numpy"),
)

py_library(
    name = "internal_test_harnesses",
    testonly = 1,
    srcs = ["_src/internal_test_util/test_harnesses.py"],
    visibility = [":internal"] + jax_internal_test_harnesses_visibility,
    deps = [
        ":jax",
    ] + py_deps("numpy"),
)

py_library(
    name = "internal_export_back_compat_test_util",
    testonly = 1,
    srcs = ["_src/internal_test_util/export_back_compat_test_util.py"],
    visibility = [
        ":internal",
    ] + jax_internal_export_back_compat_test_util_visibility,
    deps = [
        ":jax",
        "//jax/experimental/export",
    ] + py_deps("numpy"),
)

py_library(
    name = "internal_export_back_compat_test_data",
    testonly = 1,
    srcs = glob(["_src/internal_test_util/export_back_compat_test_data/*.py"]),
    visibility = [
        ":internal",
    ],
    deps = py_deps("numpy"),
)

py_library_providing_imports_info(
    name = "jax",
    srcs = [
        "_src/__init__.py",
        "_src/ad_checkpoint.py",
        "_src/api.py",
        "_src/array.py",
        "_src/callback.py",
        "_src/checkify.py",
        "_src/custom_batching.py",
        "_src/custom_derivatives.py",
        "_src/custom_transpose.py",
        "_src/debugging.py",
        "_src/dispatch.py",
        "_src/dlpack.py",
        "_src/flatten_util.py",
        "_src/interpreters/__init__.py",
        "_src/interpreters/ad.py",
        "_src/interpreters/batching.py",
        "_src/interpreters/pxla.py",
        "_src/maps.py",
        "_src/pjit.py",
        "_src/prng.py",
        "_src/public_test_util.py",
        "_src/random.py",
        "_src/shard_alike.py",
        "_src/stages.py",
        "_src/tree.py",
    ] + glob(
        [
            "*.py",
            "_src/debugger/**/*.py",
            "_src/extend/**/*.py",
            "_src/image/**/*.py",
            "_src/lax/**/*.py",
            "_src/nn/**/*.py",
            "_src/numpy/**/*.py",
            "_src/ops/**/*.py",
            "_src/scipy/**/*.py",
            "_src/state/**/*.py",
            "_src/third_party/**/*.py",
            "experimental/key_reuse/**/*.py",
            "image/**/*.py",
            "interpreters/**/*.py",
            "lax/**/*.py",
            "lib/**/*.py",
            "nn/**/*.py",
            "numpy/**/*.py",
            "ops/**/*.py",
            "scipy/**/*.py",
            "third_party/**/*.py",
        ],
        exclude = [
            "*_test.py",
            "**/*_test.py",
            "_src/internal_test_util/**",
        ],
    ) + [
        "experimental/attrs.py",
        # until new parallelism APIs are moved out of experimental
        "experimental/maps.py",
        "experimental/pjit.py",
        "experimental/multihost_utils.py",
        "experimental/shard_map.py",
        # until checkify is moved out of experimental
        "experimental/checkify.py",
        "experimental/compilation_cache/compilation_cache.py",
    ],
    lib_rule = pytype_library,
    pytype_srcs = glob(
        [
            "numpy/*.pyi",
            "_src/**/*.pyi",
        ],
        exclude = [
            "_src/basearray.pyi",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":abstract_arrays",
        ":ad_util",
        ":api_util",
        ":basearray",
        ":cloud_tpu_init",
        ":compilation_cache_internal",
        ":compiler",
        ":config",
        ":core",
        ":custom_api_util",
        ":deprecations",
        ":dtypes",
        ":effects",
        ":environment_info",
        ":jaxpr_util",
        ":layout",
        ":lazy_loader",
        ":mesh",
        ":mlir",
        ":monitoring",
        ":op_shardings",
        ":partial_eval",
        ":partition_spec",
        ":path",
        ":pickle_util",
        ":pretty_printer",
        ":profiler",
        ":sharding",
        ":sharding_impls",
        ":sharding_specs",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
        ":version",
        ":xla",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy") + py_deps("scipy") + py_deps("opt_einsum") + jax_extra_deps,
)

pytype_strict_library(
    name = "abstract_arrays",
    srcs = ["_src/abstract_arrays.py"],
    deps = [
        ":ad_util",
        ":core",
        ":dtypes",
        ":traceback_util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "ad_util",
    srcs = ["_src/ad_util.py"],
    deps = [
        ":core",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
    ],
)

pytype_strict_library(
    name = "api_util",
    srcs = ["_src/api_util.py"],
    deps = [
        ":abstract_arrays",
        ":config",
        ":core",
        ":dtypes",
        ":traceback_util",
        ":tree_util",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "basearray",
    srcs = ["_src/basearray.py"],
    pytype_srcs = ["_src/basearray.pyi"],
    deps = [
        ":sharding",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "cloud_tpu_init",
    srcs = ["_src/cloud_tpu_init.py"],
    deps = [
        ":hardware_utils",
    ],
)

pytype_strict_library(
    name = "compilation_cache_internal",
    srcs = ["_src/compilation_cache.py"],
    visibility = [":internal"] + jax_visibility("compilation_cache"),
    deps = [
        ":cache_key",
        ":compilation_cache_interface",
        ":config",
        ":gfile_cache",
        ":monitoring",
        ":path",
        "//jax/_src/lib",
    ] + py_deps("numpy") + py_deps("zstandard"),
)

pytype_strict_library(
    name = "cache_key",
    srcs = ["_src/cache_key.py"],
    visibility = [":internal"] + jax_visibility("compilation_cache"),
    deps = [
        ":config",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "compilation_cache_interface",
    srcs = ["_src/compilation_cache_interface.py"],
    deps = [
        ":path",
        ":util",
    ],
)

pytype_strict_library(
    name = "config",
    srcs = ["_src/config.py"],
    deps = [
        ":logging_config",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "logging_config",
    srcs = ["_src/logging_config.py"],
)

pytype_strict_library(
    name = "compiler",
    srcs = ["_src/compiler.py"],
    deps = [
        ":compilation_cache_internal",
        ":config",
        ":mlir",
        ":monitoring",
        ":profiler",
        ":traceback_util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "core",
    srcs = [
        "_src/core.py",
        "_src/errors.py",
        "_src/linear_util.py",
    ],
    deps = [
        ":config",
        ":dtypes",
        ":effects",
        ":pretty_printer",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "custom_api_util",
    srcs = ["_src/custom_api_util.py"],
)

pytype_strict_library(
    name = "deprecations",
    srcs = ["_src/deprecations.py"],
)

pytype_strict_library(
    name = "dtypes",
    srcs = [
        "_src/dtypes.py",
    ],
    deps = [
        ":config",
        ":traceback_util",
        ":typing",
        ":util",
    ] + py_deps("ml_dtypes") + py_deps("numpy"),
)

pytype_strict_library(
    name = "effects",
    srcs = ["_src/effects.py"],
)

pytype_strict_library(
    name = "environment_info",
    srcs = ["_src/environment_info.py"],
    deps = [
        ":version",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "gfile_cache",
    srcs = ["_src/gfile_cache.py"],
    deps = [
        ":compilation_cache_interface",
        ":path",
    ],
)

pytype_strict_library(
    name = "hardware_utils",
    srcs = ["_src/hardware_utils.py"],
)

pytype_library(
    name = "lax_reference",
    srcs = ["_src/lax_reference.py"],
    visibility = [":internal"] + jax_visibility("lax_reference"),
    deps = [
        ":core",
        ":util",
    ] + py_deps("numpy") + py_deps("scipy") + py_deps("opt_einsum"),
)

pytype_strict_library(
    name = "lazy_loader",
    srcs = ["_src/lazy_loader.py"],
)

pytype_strict_library(
    name = "jaxpr_util",
    srcs = ["_src/jaxpr_util.py"],
    deps = [
        ":core",
        ":source_info_util",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "mesh",
    srcs = ["_src/mesh.py"],
    deps = [
        ":config",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "mlir",
    srcs = ["_src/interpreters/mlir.py"],
    deps = [
        ":ad_util",
        ":config",
        ":core",
        ":dtypes",
        ":effects",
        ":layout",
        ":op_shardings",
        ":partial_eval",
        ":path",
        ":pickle_util",
        ":sharding_impls",
        ":source_info_util",
        ":state_types",
        ":util",
        ":xla",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "monitoring",
    srcs = ["_src/monitoring.py"],
)

pytype_strict_library(
    name = "op_shardings",
    srcs = ["_src/op_shardings.py"],
    deps = [
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas",
    srcs = glob(
        [
            "experimental/pallas/**/*.py",
        ],
        exclude = [
            "experimental/pallas/gpu.py",
            "experimental/pallas/tpu.py",
            "experimental/pallas/ops/tpu/**/*.py",
        ],
    ),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":jax",
        ":source_info_util",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_tpu",
    srcs = ["experimental/pallas/tpu.py"],
    visibility = [
        ":pallas_tpu_users",
    ],
    deps = [
        ":pallas",  # buildcleaner: keep
        ":tpu_custom_call",
        "//jax/_src/pallas/mosaic",
    ],
)

pytype_strict_library(
    name = "pallas_gpu_ops",
    srcs = glob(["experimental/pallas/ops/gpu/**/*.py"]),
    visibility = [
        ":pallas_gpu_users",
    ],
    deps = [
        ":jax",
        ":pallas",
        ":pallas_gpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_tpu_ops",
    srcs = glob(["experimental/pallas/ops/tpu/**/*.py"]),
    visibility = [
        ":pallas_tpu_users",
    ],
    deps = [
        ":jax",
        ":pallas",
        ":pallas_tpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_gpu",
    srcs = ["experimental/pallas/gpu.py"],
    visibility = [
        ":pallas_gpu_users",
    ],
    deps = [
        ":pallas",
        "//jax/_src/pallas/triton",
    ],
)

pytype_strict_library(
    name = "partial_eval",
    srcs = ["_src/interpreters/partial_eval.py"],
    deps = [
        ":ad_util",
        ":api_util",
        ":config",
        ":core",
        ":dtypes",
        ":effects",
        ":profiler",
        ":source_info_util",
        ":state_types",
        ":tree_util",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "partition_spec",
    srcs = ["_src/partition_spec.py"],
)

pytype_strict_library(
    name = "path",
    srcs = ["_src/path.py"],
    deps = py_deps("epath"),
)

pytype_library(
    name = "experimental_profiler",
    srcs = ["experimental/profiler.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "pickle_util",
    srcs = ["_src/pickle_util.py"],
    deps = [":profiler"] + py_deps("cloudpickle"),
)

pytype_strict_library(
    name = "pretty_printer",
    srcs = ["_src/pretty_printer.py"],
    deps = [
        ":config",
        ":util",
    ] + py_deps("colorama"),
)

pytype_strict_library(
    name = "profiler",
    srcs = ["_src/profiler.py"],
    deps = [
        ":traceback_util",
        ":xla_bridge",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "sharding",
    srcs = ["_src/sharding.py"],
    deps = [
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "layout",
    srcs = ["_src/layout.py"],
    deps = [
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "sharding_impls",
    srcs = ["_src/sharding_impls.py"],
    deps = [
        ":config",
        ":mesh",
        ":op_shardings",
        ":partition_spec",
        ":sharding",
        ":sharding_specs",
        ":tree_util",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "sharding_specs",
    srcs = ["_src/sharding_specs.py"],
    deps = [
        ":config",
        ":op_shardings",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "source_info_util",
    srcs = ["_src/source_info_util.py"],
    visibility = [":internal"] + jax_visibility("source_info_util"),
    deps = [
        ":traceback_util",
        ":version",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "state_types",
    srcs = [
        "_src/state/__init__.py",
        "_src/state/indexing.py",
        "_src/state/types.py",
    ],
    deps = [
        ":core",
        ":effects",
        ":pretty_printer",
        ":tree_util",
        ":typing",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "tree_util",
    srcs = ["_src/tree_util.py"],
    visibility = [":internal"] + jax_visibility("tree_util"),
    deps = [
        ":traceback_util",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "traceback_util",
    srcs = ["_src/traceback_util.py"],
    visibility = [":internal"] + jax_visibility("traceback_util"),
    deps = [
        ":config",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "typing",
    srcs = [
        "_src/typing.py",
    ],
    deps = [":basearray"] + py_deps("numpy"),
)

pytype_strict_library(
    name = "tpu_custom_call",
    srcs = ["_src/tpu_custom_call.py"],
    visibility = [":internal"],
    deps = [
        ":config",
        ":core",
        ":jax",
        ":mlir",
        ":sharding_impls",
        "//jax/_src/lib",
    ] + if_building_jaxlib([
        "//jaxlib/mlir:ir",
        "//jaxlib/mlir:stablehlo_dialect",
    ]) + py_deps("numpy") + py_deps("absl/flags"),
)

pytype_strict_library(
    name = "util",
    srcs = ["_src/util.py"],
    deps = [
        ":config",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "version",
    srcs = ["version.py"],
)

pytype_strict_library(
    name = "xla",
    srcs = ["_src/interpreters/xla.py"],
    deps = [
        ":abstract_arrays",
        ":config",
        ":core",
        ":dtypes",
        ":sharding_impls",
        ":source_info_util",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

# TODO(phawkins): break up this SCC.
pytype_strict_library(
    name = "xla_bridge",
    srcs = [
        "_src/clusters/__init__.py",
        "_src/clusters/cloud_tpu_cluster.py",
        "_src/clusters/cluster.py",
        "_src/clusters/ompi_cluster.py",
        "_src/clusters/slurm_cluster.py",
        "_src/distributed.py",
        "_src/xla_bridge.py",
    ],
    visibility = [":internal"] + jax_visibility("xla_bridge"),
    deps = [
        ":cloud_tpu_init",
        ":config",
        ":hardware_utils",
        ":traceback_util",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("importlib_metadata"),
)

# Public JAX libraries below this point.

py_library_providing_imports_info(
    name = "experimental",
    srcs = glob(
        [
            "experimental/*.py",
            "example_libraries/*.py",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
    ] + py_deps("absl/logging") + py_deps("numpy"),
)

pytype_library(
    name = "stax",
    srcs = [
        "example_libraries/stax.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "experimental_array_api",
    srcs = glob(
        [
            "experimental/array_api/*.py",
        ],
    ),
    visibility = [":internal"],
    deps = [":jax"],
)

pytype_library(
    name = "experimental_sparse",
    srcs = glob(
        [
            "experimental/sparse/*.py",
        ],
        exclude = ["experimental/sparse/test_util.py"],
    ),
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "sparse_test_util",
    testonly = 1,
    srcs = [
        "experimental/sparse/test_util.py",
    ],
    visibility = [":internal"],
    deps = [
        ":experimental_sparse",
        ":jax",
        ":test_util",
    ] + py_deps("numpy"),
)

pytype_library(
    name = "optimizers",
    srcs = [
        "example_libraries/optimizers.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"] + py_deps("numpy"),
)

pytype_library(
    name = "ode",
    srcs = ["experimental/ode.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

# TODO(apaszke): Remove this target
pytype_library(
    name = "maps",
    srcs = ["experimental/maps.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        ":mesh",
    ],
)

# TODO(apaszke): Remove this target
pytype_library(
    name = "pjit",
    srcs = ["experimental/pjit.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":experimental",
        ":jax",
    ],
)

pytype_library(
    name = "jet",
    srcs = ["experimental/jet.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "experimental_host_callback",
    srcs = ["experimental/host_callback.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
    ],
)

pytype_library(
    name = "compilation_cache",
    srcs = [
        "experimental/compilation_cache/compilation_cache.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "mesh_utils",
    srcs = ["experimental/mesh_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_bridge",
    ],
)

# TODO(phawkins): remove this target in favor of the finer-grained targets in jax/extend/...
pytype_strict_library(
    name = "extend",
    visibility = [":jax_extend_users"],
    deps = [
        "//jax/extend",
        "//jax/extend:core",
        "//jax/extend:linear_util",
        "//jax/extend:random",
        "//jax/extend:source_info_util",
    ],
)

pytype_library(
    name = "mosaic",
    srcs = [
        "experimental/mosaic/__init__.py",
        "experimental/mosaic/dialects.py",
    ],
    visibility = [":mosaic_users"],
    deps = [
        ":tpu_custom_call",
        "//jax/_src/lib",
    ],
)

pytype_library(
    name = "rnn",
    srcs = ["experimental/rnn.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "fused_attention_stablehlo",
    srcs = [
        "_src/cudnn/__init__.py",
        "_src/cudnn/fused_attention_stablehlo.py",
    ],
    visibility = [":internal"],
    deps = [
        ":experimental",
        ":jax",
    ],
)
