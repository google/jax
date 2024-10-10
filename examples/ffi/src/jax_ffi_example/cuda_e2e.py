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
"""An end-to-end example demonstrating the use of the JAX FFI with CUDA.

The specifics of the kernels are not very important, but the general structure,
and packaging of the extension are useful for testing.
"""

import os
import ctypes

import numpy as np

import jax
import jax.numpy as jnp
import jax.extend as jex

# Load the shared library with the FFI target definitions
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "lib_cuda_e2e.so")
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

jex.ffi.register_ffi_target("foo-fwd", jex.ffi.pycapsule(library.FooFwd),
                            platform="CUDA")
jex.ffi.register_ffi_target("foo-bwd", jex.ffi.pycapsule(library.FooBwd),
                            platform="CUDA")


def foo_fwd(a, b):
  assert a.dtype == jnp.float32
  assert a.shape == b.shape
  assert a.dtype == b.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  c, b_plus_1 = jex.ffi.ffi_call("foo-fwd", (out_type, out_type), a, b, n=n)
  return c, (a, b_plus_1)


def foo_bwd(res, c_grad):
  a, b_plus_1 = res
  assert c_grad.dtype == jnp.float32
  assert c_grad.shape == a.shape
  assert a.shape == b_plus_1.shape
  assert c_grad.dtype == a.dtype
  assert a.dtype == b_plus_1.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  return jex.ffi.ffi_call("foo-bwd", (out_type, out_type), c_grad, a, b_plus_1,
                          n=n)


@jax.custom_vjp
def foo(a, b):
  c, _ = foo_fwd(a, b)
  return c


foo.defvjp(foo_fwd, foo_bwd)
