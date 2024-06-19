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
    "cuda_library",
    "jax_test",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_test(
    name = "cuda_custom_call_test",
    srcs = ["cuda_custom_call_test.py"],
    # libfoo.so is a runtime dependency for this test
    data = [":foo"],
    disable_backends = [
        "cpu",
        "tpu",
    ],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax:extend",
    ],
)

# this second target is needed to properly link in CUDA runtime symbols
# such as cudaLaunchKernel, even though we are only building one library.
cc_shared_library(
    name = "foo",
    deps = [
        ":foo_",
        "@xla//xla/tsl/cuda:cudart",
    ],
)

cuda_library(
    name = "foo_",
    srcs = ["foo.cu.cc"],
    deps = [
        "@xla//xla/ffi/api",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)
