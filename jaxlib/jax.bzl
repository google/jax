# Copyright 2021 The JAX Authors.
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

"""Bazel macros used by the JAX build."""

load("@com_github_google_flatbuffers//:build_defs.bzl", _flatbuffer_cc_library = "flatbuffer_cc_library")
load("@local_config_cuda//cuda:build_defs.bzl", _cuda_library = "cuda_library", _if_cuda_is_configured = "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", _if_rocm_is_configured = "if_rocm_is_configured", _rocm_library = "rocm_library")
load("@rules_cc//cc:defs.bzl", _cc_proto_library = "cc_proto_library")
load("@tsl//tsl:tsl.bzl", _if_windows = "if_windows", _pybind_extension = "tsl_pybind_extension_opensource")
load("@tsl//tsl/platform:build_config_root.bzl", _tf_cuda_tests_tags = "tf_cuda_tests_tags", _tf_exec_properties = "tf_exec_properties")

# Explicitly re-exports names to avoid "unused variable" warnings from .bzl
# lint tools.
cc_proto_library = _cc_proto_library
cuda_library = _cuda_library
rocm_library = _rocm_library
pytype_test = native.py_test
pybind_extension = _pybind_extension
if_cuda_is_configured = _if_cuda_is_configured
if_rocm_is_configured = _if_rocm_is_configured
if_windows = _if_windows
flatbuffer_cc_library = _flatbuffer_cc_library
tf_exec_properties = _tf_exec_properties
tf_cuda_tests_tags = _tf_cuda_tests_tags

jax_internal_packages = []
jax_extend_internal_users = []
mosaic_internal_users = []
pallas_gpu_internal_users = []
pallas_tpu_internal_users = []

jax_internal_export_back_compat_test_util_visibility = []
jax_internal_test_harnesses_visibility = []
jax_test_util_visibility = []
loops_visibility = []

def py_deps(_package):
    """Returns the Bazel deps for Python package `package`."""

    # We assume the user has installed all dependencies in their Python environment.
    # This indirection exists because in Google's internal build we build
    # dependencies from source with Bazel, but that's not something most people would want.
    return []

def jax_visibility(_target):
    """Returns the additional Bazel visibilities for `target`."""

    # This is only useful as part of a larger Bazel repository.
    return []

jax_extra_deps = []
jax2tf_deps = []

def pytype_library(name, pytype_srcs = None, **kwargs):
    _ = pytype_srcs  # @unused
    native.py_library(name = name, **kwargs)

def pytype_strict_library(name, pytype_srcs = None, **kwargs):
    _ = pytype_srcs  # @unused
    native.py_library(name = name, **kwargs)

def py_library_providing_imports_info(*, name, lib_rule = native.py_library, pytype_srcs = [], **kwargs):
    lib_rule(name = name, **kwargs)

def py_extension(name, srcs, copts, deps, linkopts = []):
    pybind_extension(name, srcs = srcs, copts = copts, linkopts = linkopts, deps = deps, module_name = name)

def windows_cc_shared_mlir_library(name, out, deps = [], srcs = [], exported_symbol_prefixes = []):
    """Workaround DLL building issue.

    1. cc_binary with linkshared enabled cannot produce DLL with symbol
       correctly exported.
    2. Even if the DLL is correctly built, the resulting target cannot be
       correctly consumed by other targets.

    Args:
      name: the name of the output target
      out: the name of the output DLL filename
      deps: deps
      srcs: srcs
    """

    # create a dummy library to get the *.def file
    dummy_library_name = name + ".dummy.dll"
    native.cc_binary(
        name = dummy_library_name,
        linkshared = 1,
        linkstatic = 1,
        deps = deps,
        target_compatible_with = ["@platforms//os:windows"],
    )

    # .def file with all symbols, not usable
    full_def_name = name + ".full.def"
    native.filegroup(
        name = full_def_name,
        srcs = [dummy_library_name],
        output_group = "def_file",
        target_compatible_with = ["@platforms//os:windows"],
    )

    # say filtered_symbol_prefixes == ["mlir", "chlo"], then construct the regex
    # pattern as "^\\s*(mlir|clho)" to use grep
    pattern = "^\\s*(" + "|".join(exported_symbol_prefixes) + ")"

    # filtered def_file, only the needed symbols are included
    filtered_def_name = name + ".filtered.def"
    filtered_def_file = out + ".def"
    native.genrule(
        name = filtered_def_name,
        srcs = [full_def_name],
        outs = [filtered_def_file],
        cmd = """echo 'LIBRARY {}\nEXPORTS ' > $@ && grep -E '{}' $(location :{}) >> $@""".format(out, pattern, full_def_name),
        target_compatible_with = ["@platforms//os:windows"],
    )

    # create the desired library
    native.cc_binary(
        name = out,  # this name must be correct, it will be the filename
        linkshared = 1,
        deps = deps,
        win_def_file = filtered_def_file,
        target_compatible_with = ["@platforms//os:windows"],
    )

    # however, the created cc_library (a shared library) cannot be correctly
    # consumed by other cc_*...
    interface_library_file = out + ".if.lib"
    native.filegroup(
        name = interface_library_file,
        srcs = [out],
        output_group = "interface_library",
        target_compatible_with = ["@platforms//os:windows"],
    )

    # but this one can be correctly consumed, this is our final product
    native.cc_import(
        name = name,
        interface_library = interface_library_file,
        shared_library = out,
        target_compatible_with = ["@platforms//os:windows"],
    )

ALL_BACKENDS = ["cpu", "gpu", "tpu"]

def if_building_jaxlib(if_building, if_not_building = []):
    return select({
        "//jax:enable_jaxlib_build": if_building,
        "//conditions:default": if_not_building,
    })

# buildifier: disable=function-docstring
def jax_test(
        name,
        srcs,
        args = [],
        env = {},
        shard_count = None,
        deps = [],
        disable_backends = None,  # buildifier: disable=unused-variable
        backend_variant_args = {},
        backend_tags = {},  # buildifier: disable=unused-variable
        disable_configs = None,  # buildifier: disable=unused-variable
        enable_configs = None,  # buildifier: disable=unused-variable
        config_tags_overrides = None,  # buildifier: disable=unused-variable
        tags = [],
        main = None,
        pjrt_c_api_bypass = False):  # buildifier: disable=unused-variable
    if main == None:
        if len(srcs) == 1:
            main = srcs[0]
        else:
            fail("Must set a main file to test multiple source files.")

    for backend in ALL_BACKENDS:
        if shard_count == None or type(shard_count) == type(0):
            test_shards = shard_count
        else:
            test_shards = shard_count.get(backend, 1)
        test_args = list(args) + [
            "--jax_test_dut=" + backend,
            "--jax_platform_name=" + backend,
        ]
        test_tags = list(tags) + ["jax_test_%s" % backend] + backend_tags.get(backend, [])
        if disable_backends and backend in disable_backends:
            test_tags += ["manual"]
        if backend == "gpu":
            test_tags += tf_cuda_tests_tags()
        native.py_test(
            name = name + "_" + backend,
            srcs = srcs,
            args = test_args + backend_variant_args.get(backend, []),
            env = env,
            deps = [
                "//jax",
                "//jax:test_util",
            ] + deps + if_building_jaxlib(["//jaxlib/cuda:gpu_only_test_deps"]) + select({
                "//jax:enable_build_cuda_plugin_from_source": ["//jax_plugins:gpu_plugin_only_test_deps"],
                "//conditions:default": [],
            }),
            shard_count = test_shards,
            tags = test_tags,
            main = main,
            exec_properties = tf_exec_properties({"tags": test_tags}),
        )

def jax_generate_backend_suites(backends = []):
    """Generates test suite targets named cpu_tests, gpu_tests, etc.

    Args:
      backends: the set of backends for which rules should be generated. Defaults to all backends.
    """
    if not backends:
        backends = ALL_BACKENDS
    for backend in backends:
        native.test_suite(
            name = "%s_tests" % backend,
            tags = ["jax_test_%s" % backend, "-manual"],
        )
    native.test_suite(
        name = "backend_independent_tests",
        tags = ["-jax_test_%s" % backend for backend in backends] + ["-manual"],
    )

jax_test_file_visibility = []
