# Copyright 2020 The JAX Authors.
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
"""Tests for multi-platform and cross-platform JAX export.

This module contains the tests parameterized by test_harnesses. These tests
verify that the primitive lowering rules work properly in multi-platform and
cross-platform lowering mode. The actual mechanism for multi-platform and
cross-platform lowering is tested in export_test.py.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import re

from absl import logging
from absl.testing import absltest

import numpy as np

import jax
from jax import export
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.internal_test_util import test_harnesses
from jax import random


def make_disjunction_regexp(*parts: str) -> re.Pattern[str]:
  if not parts:
    return re.compile("matches_no_test")
  else:
    return re.compile("(" + "|".join(parts) + ")")

# TODO(necula): Failures to be investigated (on GPU).
_known_failures_gpu = make_disjunction_regexp(
    # Failures on GPU due to failure to export custom call targets, these
    # involve GPU custom call targets withoutbackwards compatibility tests.
    "custom_linear_solve_",
    "lu_",
    "svd_",
    "tridiagonal_solve_",
)

# Some primitive lowering rules need the GPU backend to be able to create
# CUDA lowering.
_skip_cuda_lowering_unless_have_gpus = make_disjunction_regexp(
    "svd_", "lu_", "eigh_", "qr_", "custom_linear_", "tridiagonal_solve_",
    # TODO(b/350111820): random should work once we enable FFI threefry2x32
    "random_",
)


class PrimitiveTest(jtu.JaxTestCase):

  @classmethod
  def setUpClass(cls):
    # Pick one device from each available platform
    cls.devices = []
    cls.platforms = []
    for backend in ["cpu", "gpu", "tpu"]:
      try:
        devices = jax.devices(backend)
      except RuntimeError:
        devices = []

      for d in devices:
        if d.platform not in cls.platforms:
          cls.platforms.append(d.platform)
          cls.devices.append(d)
    super().setUpClass()

  # For each primitive we export for all platforms that are available and
  # compare the results of running the exported code and running the native
  # code.
  # If you want to run this test for only one harness, add parameter
  # `one_containing="foo"` to parameterized below.
  @test_harnesses.parameterized(
      test_harnesses.all_harnesses,
      include_jax_unimpl=False,
      # one_containing="",
  )
  @jtu.ignore_warning(
      category=UserWarning,
      message=("Using reduced precision for gradient of reduce-window min/max "
               "operator to work around missing XLA support for pair-reductions")
  )
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_prim(self, harness: test_harnesses.Harness):
    if (jtu.device_under_test() == "gpu"
        and _known_failures_gpu.search(harness.fullname)):
      self.skipTest("failure to be investigated")

    if harness.params.get("enable_xla", False):
      self.skipTest("enable_xla=False is not relevant")

    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())

    unimplemented_platforms: set[str] = set()
    for l in harness.jax_unimplemented:
      if l.filter(dtype=harness.dtype):
        unimplemented_platforms = unimplemented_platforms.union(l.devices)
    if (_skip_cuda_lowering_unless_have_gpus.search(harness.fullname)
        and all(d.platform != "gpu" for d in self.devices)):
      unimplemented_platforms.add("gpu")

    logging.info("Harness is not implemented on %s", unimplemented_platforms)

    # Tolerances.
    tol = None
    if ("conv_general_dilated" in harness.fullname
      and harness.dtype in [np.float32]):
      tol = 1e-4

    self.export_and_compare_to_native(
      func_jax, *args,
      unimplemented_platforms=unimplemented_platforms,
      tol=tol)

  def export_and_compare_to_native(
      self, func_jax: Callable,
      *args: jax.Array,
      unimplemented_platforms: set[str] = set(),
      skip_run_on_platforms: set[str] = set(),
      tol: float | None = None):
    devices = [
        d
        for d in self.__class__.devices
        if d.platform not in unimplemented_platforms
    ]
    logging.info("Using devices %s", [str(d) for d in devices])
    # lowering_platforms uses "cuda" or "rocm" instead of "gpu"
    gpu_platform = "cuda"
    if jtu.is_device_rocm():
        gpu_platform = "rocm"
    lowering_platforms: list[str] = [
        p if p != "gpu" else gpu_platform
        for p in ("cpu", "gpu", "tpu")
        if p not in unimplemented_platforms
    ]

    if len(lowering_platforms) <= 1:
      self.skipTest(
          "Harness is uninteresting with fewer than 2 platforms"
      )

    logging.info("Exporting harness for %s", lowering_platforms)
    exp = export.export(jax.jit(func_jax),
                        lowering_platforms=lowering_platforms)(*args)

    for device in devices:
      if device.platform in skip_run_on_platforms:
        logging.info("Skipping running on %s", device)
        continue
      device_args = jax.tree.map(
          lambda x: jax.device_put(x, device), args
      )
      logging.info("Running harness natively on %s", device)
      native_res = func_jax(*device_args)
      logging.info("Running exported harness on %s", device)
      exported_res = exp.call(*device_args)
      if tol is not None:
        logging.info(f"Using non-standard tolerance {tol}")
      self.assertAllClose(native_res, exported_res, atol=tol, rtol=tol)
      # TODO(necula): Check HLO equivalence for the ultimate test.

  def test_psum_scatter(self):
    f = jax.jit(jax.pmap(lambda x: lax.psum_scatter(x, 'i'),
                         axis_name='i',
                         devices=jax.devices()[:1]))

    shape = (1, 1, 8)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    self.export_and_compare_to_native(f, x)

  # The lowering rule for all_gather has special cases for bool.
  @jtu.parameterized_filterable(
    kwargs=[
      dict(dtype=dtype)
      for dtype in [np.bool_, np.float32]],
  )
  def test_all_gather(self, *, dtype):
    f = jax.jit(jax.pmap(lambda x: lax.all_gather(x, 'i'),
                         axis_name='i',
                         devices=jax.devices()[:1]))

    shape = (1, 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    if dtype == np.bool_:
      x = (x % 2).astype(np.bool_)
    self.export_and_compare_to_native(f, x)

  def test_random_with_threefry_gpu_kernel_lowering(self):
    # On GPU we use a custom call for threefry2x32
    with config.threefry_gpu_kernel_lowering(True):
      def f(x):
        return random.gamma(random.key(42), x)

      shape = (4, 5)
      x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
      self.export_and_compare_to_native(f, x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
