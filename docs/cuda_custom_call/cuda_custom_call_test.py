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


# This test is intentionally structured to stay close to what a standalone JAX
# custom call integration might look like. JAX test harness is in a separate
# section towards the end of this file. The test can be run standalone by typing
# "make" in the directory containing this file.

import os
import ctypes
import unittest

import numpy as np

import jax
import jax.numpy as jnp
from jax.extend import ffi
from jax.lib import xla_client

# start test boilerplate
from absl.testing import absltest
from jax._src import config
from jax._src import test_util as jtu

config.parse_flags_with_absl()
# end test boilerplate

# XLA needs uppercase, "cuda" isn't recognized
XLA_PLATFORM = "CUDA"

# JAX needs lowercase, "CUDA" isn't recognized
JAX_PLATFORM = "cuda"

# 0 = original ("opaque"), 1 = FFI
XLA_CUSTOM_CALL_API_VERSION = 1

# these strings are how we identify kernels to XLA:
# - first we register a pointer to the kernel with XLA under this name
# - then we "tell" JAX to emit StableHLO specifying this name to XLA
XLA_CUSTOM_CALL_TARGET_FWD = "foo-fwd"
XLA_CUSTOM_CALL_TARGET_BWD = "foo-bwd"

# load the shared library with the FFI target definitions
if jtu.is_running_under_pytest():
  raise unittest.SkipTest("libfoo.so hasn't been built")
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "libfoo.so")
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

# register the custom calls targets with XLA
xla_client.register_custom_call_target(name=XLA_CUSTOM_CALL_TARGET_FWD,
                                       fn=ffi.pycapsule(library.FooFwd),
                                       platform=XLA_PLATFORM,
                                       api_version=XLA_CUSTOM_CALL_API_VERSION)
xla_client.register_custom_call_target(name=XLA_CUSTOM_CALL_TARGET_BWD,
                                       fn=ffi.pycapsule(library.FooBwd),
                                       platform=XLA_PLATFORM,
                                       api_version=XLA_CUSTOM_CALL_API_VERSION)


def foo_fwd(a, b):
  assert a.dtype == jnp.float32
  assert a.shape == b.shape
  assert a.dtype == b.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  c, b_plus_1 = ffi.ffi_call(XLA_CUSTOM_CALL_TARGET_FWD, (out_type, out_type),
                             a, b, n=n)
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
  return ffi.ffi_call(XLA_CUSTOM_CALL_TARGET_BWD, (out_type, out_type),
                      c_grad, a, b_plus_1, n=n)


@jax.custom_vjp
def foo(a, b):
  c, _ = foo_fwd(a, b)
  return c


foo.defvjp(foo_fwd, foo_bwd)

#-----------------------------------------------------------------------------#
#                                  Test                                       #
#-----------------------------------------------------------------------------#


class CustomCallTest(jtu.JaxTestCase):

  def test_fwd_interpretable(self):
    shape = (2, 3)
    a = 2. * jnp.ones(shape)
    b = 3. * jnp.ones(shape)
    observed = jax.jit(foo)(a, b)
    expected = (2. * (3. + 1.))
    self.assertArraysEqual(observed, expected)

  def test_bwd_interpretable(self):
    shape = (2, 3)
    a = 2. * jnp.ones(shape)
    b = 3. * jnp.ones(shape)

    def loss(a, b):
      return jnp.sum(foo(a, b))

    da_observed, db_observed = jax.jit(jax.grad(loss, argnums=(0, 1)))(a, b)
    da_expected = b + 1
    db_expected = a
    self.assertArraysEqual(da_observed, da_expected)
    self.assertArraysEqual(db_observed, db_expected)

  def test_fwd_random(self):
    shape = (2, 3)
    akey, bkey = jax.random.split(jax.random.key(0))
    a = jax.random.normal(key=akey, shape=shape)
    b = jax.random.normal(key=bkey, shape=shape)
    observed = jax.jit(foo)(a, b)
    expected = a * (b + 1)
    self.assertAllClose(observed, expected)

  def test_bwd_random(self):
    shape = (2, 3)
    akey, bkey = jax.random.split(jax.random.key(0))
    a = jax.random.normal(key=akey, shape=shape)
    b = jax.random.normal(key=bkey, shape=shape)
    jtu.check_grads(f=jax.jit(foo), args=(a, b), order=1, modes=("rev",))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
