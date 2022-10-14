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
    "jax_extra_deps",
    "jax_internal_packages",
    "jax_test_util_visibility",
    "py_deps",
    "py_library_providing_imports_info",
    "pytype_library",
)

licenses(["notice"])

package(default_visibility = [":internal"])

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

py_library_providing_imports_info(
    name = "jax",
    srcs = glob(
        [
            "*.py",
            "_src/**/*.py",
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
            "_src/test_util.py",
            "*_test.py",
            "**/*_test.py",
        ],
    ) + [
        # until new parallelism APIs are moved out of experimental
        "experimental/maps.py",
        "experimental/pjit.py",
        "experimental/global_device_array.py",
        "experimental/multihost_utils.py",
        # until checkify is moved out of experimental
        "experimental/checkify.py",
        # to avoid circular dependencies
        "experimental/compilation_cache/compilation_cache.py",
        "experimental/compilation_cache/gfile_cache.py",
        "experimental/compilation_cache/cache_interface.py",
    ],
    lib_rule = pytype_library,
    pytype_srcs = glob(["_src/**/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = select({
               ":enable_jaxlib_build": [":jaxlib_deps"],
               "//conditions:default": [],
           }) +
           py_deps("numpy") + py_deps("scipy") + jax_extra_deps,
)

py_library(
    name = "jaxlib_deps",
    deps = [
        "//jaxlib",
    ],
)

py_library_providing_imports_info(
    name = "experimental",
    srcs = glob([
        "experimental/*.py",
        "example_libraries/*.py",
    ]),
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
    name = "experimental_sparse",
    srcs = glob([
        "experimental/sparse/*.py",
    ]),
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "optimizers",
    srcs = [
        "example_libraries/optimizers.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "ode",
    srcs = ["experimental/ode.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "callback",
    srcs = ["experimental/callback.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

# TODO(apaszke): Remove this target
pytype_library(
    name = "maps",
    srcs = ["experimental/maps.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
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
        "experimental/compilation_cache/gfile_cache.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "mesh_utils",
    srcs = ["experimental/mesh_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":experimental",
        ":jax",
    ],
)

pytype_library(
    name = "experimental_program_eviction",
    srcs = ["experimental/program_eviction.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
    ],
)
