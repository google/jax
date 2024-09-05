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
"""Tests for backwards compatibility of exporting code with custom calls.

See the export_back_compat_test_util module docstring for how to setup and
update these tests.
"""
import dataclasses
from functools import partial
import itertools
import math

from absl.testing import absltest, parameterized

import numpy as np

import jax
from jax import lax
from jax._src.export import _export

from jax._src.internal_test_util import export_back_compat_test_util as bctu

from jax._src.internal_test_util.export_back_compat_test_data import cpu_cholesky_lapack_potrf
from jax._src.internal_test_util.export_back_compat_test_data import cpu_eig_lapack_geev
from jax._src.internal_test_util.export_back_compat_test_data import cuda_eigh_cusolver_syev
from jax._src.internal_test_util.export_back_compat_test_data import rocm_eigh_hipsolver_syev
from jax._src.internal_test_util.export_back_compat_test_data import cpu_eigh_lapack_syev
from jax._src.internal_test_util.export_back_compat_test_data import cpu_lu_lapack_getrf
from jax._src.internal_test_util.export_back_compat_test_data import cuda_qr_cusolver_geqrf
from jax._src.internal_test_util.export_back_compat_test_data import rocm_qr_hipsolver_geqrf
from jax._src.internal_test_util.export_back_compat_test_data import cpu_qr_lapack_geqrf
from jax._src.internal_test_util.export_back_compat_test_data import cpu_schur_lapack_gees
from jax._src.internal_test_util.export_back_compat_test_data import cpu_svd_lapack_gesdd
from jax._src.internal_test_util.export_back_compat_test_data import cpu_triangular_solve_blas_trsm
from jax._src.internal_test_util.export_back_compat_test_data import cpu_tridiagonal_lapack_sytrd_hetrd
from jax._src.internal_test_util.export_back_compat_test_data import cuda_threefry2x32
from jax._src.internal_test_util.export_back_compat_test_data import cuda_lu_pivots_to_permutation
from jax._src.internal_test_util.export_back_compat_test_data import cuda_lu_cusolver_getrf
from jax._src.internal_test_util.export_back_compat_test_data import tpu_Eigh
from jax._src.internal_test_util.export_back_compat_test_data import tpu_Lu
from jax._src.internal_test_util.export_back_compat_test_data import tpu_ApproxTopK
from jax._src.internal_test_util.export_back_compat_test_data import tpu_Qr
from jax._src.internal_test_util.export_back_compat_test_data import tpu_Sharding
from jax._src.internal_test_util.export_back_compat_test_data import tpu_stablehlo_dynamic_reduce_window
from jax._src.internal_test_util.export_back_compat_test_data import stablehlo_dynamic_rng_bit_generator
from jax._src.internal_test_util.export_back_compat_test_data import stablehlo_dynamic_top_k
from jax._src.internal_test_util.export_back_compat_test_data import stablehlo_dynamic_approx_top_k

from jax.experimental import pjit
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import cuda_versions
from jax._src.lib import version as jaxlib_version

config.parse_flags_with_absl()


def _is_required_cusolver_version_satisfied(required_version):
  if cuda_versions is None:
    return False
  return cuda_versions.cusolver_get_version() >= required_version


@jtu.with_config(jax_legacy_prng_key="allow",
                 jax_debug_key_reuse=False,
                 jax_include_full_tracebacks_in_locations=False,
                 jax_threefry_gpu_kernel_lowering=True)
class CompatTest(bctu.CompatTestBase):
  def test_dummy(self):
    # Tests the testing mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(bctu.dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data, platform=self.default_jax_backend())
    self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_output(self):
    # Test the detection mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(bctu.dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=self.default_jax_backend(),
        expected_outputs=(np.array(2.0, dtype=np.float32),))
    with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_custom_calls(self):
    # Test the detection mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(bctu.dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=self.default_jax_backend(),
        custom_call_targets=["missing"])
    with self.assertRaisesRegex(AssertionError, "Element counts were not equal"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_custom_call_coverage(self):
    """Tests that the back compat tests cover all the targets declared stable."""
    targets_to_cover = set(_export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE)
    cpu_ffi_testdatas = [
        cpu_cholesky_lapack_potrf.data_2024_05_31,
        cpu_qr_lapack_geqrf.data_2024_08_22,
        cpu_eig_lapack_geev.data_2024_08_19,
        cpu_eigh_lapack_syev.data_2024_08_19,
        cpu_lu_lapack_getrf.data_2024_05_31,
        cpu_svd_lapack_gesdd.data_2024_08_13,
    ]
    # Add here all the testdatas that should cover the targets guaranteed
    # stable
    covering_testdatas = [
        *cpu_ffi_testdatas,
        cpu_cholesky_lapack_potrf.data_2023_06_19,
        cpu_eig_lapack_geev.data_2023_06_19,
        cpu_eigh_lapack_syev.data_2023_03_17,
        cpu_qr_lapack_geqrf.data_2023_03_17,
        cuda_threefry2x32.data_2023_03_15, cuda_threefry2x32.data_2024_07_30,
        cpu_lu_lapack_getrf.data_2023_06_14,
        cuda_lu_pivots_to_permutation.data_2024_08_08,
        cuda_lu_cusolver_getrf.data_2024_08_19,
        cuda_qr_cusolver_geqrf.data_2023_03_18,
        cuda_eigh_cusolver_syev.data_2023_03_17,
        rocm_qr_hipsolver_geqrf.data_2024_08_05,
        rocm_eigh_hipsolver_syev.data_2024_08_05,
        cpu_schur_lapack_gees.data_2023_07_16,
        cpu_svd_lapack_gesdd.data_2023_06_19,
        cpu_triangular_solve_blas_trsm.data_2023_07_16,
        tpu_Eigh.data, tpu_Lu.data_2023_03_21, tpu_Qr.data_2023_03_17,
        tpu_Sharding.data_2023_03_16, tpu_ApproxTopK.data_2023_04_17,
        tpu_ApproxTopK.data_2023_05_16,
        tpu_stablehlo_dynamic_reduce_window.data_unary_2023_06_17,
        tpu_stablehlo_dynamic_reduce_window.data_variadic_2023_06_17,
        stablehlo_dynamic_rng_bit_generator.data_2023_06_17,
        stablehlo_dynamic_top_k.data_2023_07_16,
        stablehlo_dynamic_top_k.data_2023_08_11,  # with shape_assertion
        stablehlo_dynamic_approx_top_k.data_2024_05_30,
    ]
    # Some of the above are nested structures.
    covering_testdatas = itertools.chain(
        *[self.load_testdata_nested(d) for d in covering_testdatas])
    covered_targets = set()
    for data in covering_testdatas:
      self.assertIsInstance(data, bctu.CompatTestData)
      covered_targets = covered_targets.union(data.custom_call_targets)

    covered_targets = covered_targets.union({
      "tf.call_tf_function",  # tested in jax2tf/tests/back_compat_tf_test.py
      "tpu_custom_call",  # tested separately
      "__gpu$xla.gpu.triton",  # tested in pallas/export_back_compat_pallas_test.py
    })
    not_covered = targets_to_cover.difference(covered_targets)
    self.assertEmpty(not_covered,
                     msg=("The following custom call targets are declared "
                          "stable but are not covered by any tests: "
                          f"{not_covered}"))

  def cholesky_input(self, shape, dtype):
    a = jtu.rand_default(self.rng())(shape, dtype)
    return np.matmul(a, np.conj(np.swapaxes(a, -1, -2)))

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_cholesky_lapack_potrf(self, dtype_name="f32"):
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    shape = (4, 4)
    input = self.cholesky_input(shape, dtype)
    del input  # Input is in the testdata, here for readability
    func = lax.linalg.cholesky

    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]

    data = self.load_testdata(cpu_cholesky_lapack_potrf.data_2023_06_19[dtype_name])
    # TODO(b/344892332): Remove the check after the compatibility period.
    has_xla_ffi_support = jaxlib_version >= (0, 4, 31)
    self.run_one_test(func, data, rtol=rtol, atol=atol)
    if has_xla_ffi_support:
      with config.export_ignore_forward_compatibility(True):
        # FFI Kernel test
        data = self.load_testdata(cpu_cholesky_lapack_potrf.data_2024_05_31[dtype_name])
        self.run_one_test(func, data, rtol=rtol, atol=atol)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_eig_lapack_geev(self, dtype_name="f32"):
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    shape = (4, 4)
    def func():
      # Compute the inputs to simplify the harness
      input = jnp.arange(math.prod(shape), dtype=dtype).reshape(shape)
      return lax.linalg.eig(input,
                            compute_left_eigenvectors=True,
                            compute_right_eigenvectors=True)

    data = self.load_testdata(cpu_eig_lapack_geev.data_2023_06_19[dtype_name])
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]

    def check_eig_results(res_run, res_expected, *, rtol, atol):
      # Test ported from tests.linlag_test.testEig
      # Norm, adjusted for dimension and type.
      inner_dimension = shape[-1]
      operand = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
      def norm(x):
        norm = np.linalg.norm(x, axis=(-2, -1))
        return norm / ((inner_dimension + 1) * jnp.finfo(dtype).eps)

      def check_right_eigenvectors(a, w, vr):
        self.assertTrue(
            np.all(norm(np.matmul(a, vr) - w[..., None, :] * vr) < 100))

      def check_left_eigenvectors(a, w, vl):
        rank = len(a.shape)
        aH = jnp.conj(a.transpose(list(range(rank - 2)) + [rank - 1, rank - 2]))
        wC = jnp.conj(w)
        check_right_eigenvectors(aH, wC, vl)

      def check_eigenvalue_is_in_array(eigenvalue, eigenvalues_array):
        closest_diff = min(abs(eigenvalues_array - eigenvalue))
        self.assertAllClose(
            closest_diff,
            np.array(0., closest_diff.dtype),
            atol=atol, rtol=rtol)

      all_w_run, all_w_exp = res_run[0], res_expected[0]
      for idx in itertools.product(*map(range, operand.shape[:-2])):
        w_run, w_exp = all_w_run[idx], all_w_exp[idx]
        for i in range(inner_dimension):
          check_eigenvalue_is_in_array(w_run[i], w_exp)
          check_eigenvalue_is_in_array(w_exp[i], w_run)

      check_left_eigenvectors(operand, all_w_run, res_run[1])
      check_right_eigenvectors(operand, all_w_run, res_run[2])

    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=check_eig_results)
    # TODO(b/344892332): Remove the check after the compatibility period.
    has_xla_ffi_support = jaxlib_version >= (0, 4, 32)
    if has_xla_ffi_support:
      with config.export_ignore_forward_compatibility(True):
        # FFI Kernel test
        data = self.load_testdata(cpu_eig_lapack_geev.data_2024_08_19[dtype_name])
        self.run_one_test(func, data, rtol=rtol, atol=atol,
                          check_results=check_eig_results)

  @staticmethod
  def eigh_input(shape, dtype):
    # In order to keep inputs small, we construct the input programmatically
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)
    # Make operand self-adjoint
    operand = (operand + jnp.conj(jnp.swapaxes(operand, -1, -2))) / 2.
    return operand

  @staticmethod
  def eigh_harness(shape, dtype):
    operand = CompatTest.eigh_input(shape, dtype)
    return lax.linalg.eigh(jnp.tril(operand), lower=True, symmetrize_input=False)

  def check_eigh_results(self, operand, res_now, res_expected, *,
                         rtol, atol=None):
    v_now, w_now = res_now
    _, w_expected = res_expected
    n, m = operand.shape
    assert n == m
    assert v_now.shape == operand.shape
    assert w_now.shape == (n,)
    self.assertLessEqual(
        np.linalg.norm(np.eye(n) - np.matmul(np.conj(np.swapaxes(v_now, -1, -2)), v_now)),
        rtol)
    # w_now : f64[n] while v_now: c128[n, n]
    w_now_like_v = w_now[np.newaxis, :].astype(v_now.dtype)
    self.assertLessEqual(
        np.linalg.norm(np.matmul(operand, v_now) - w_now_like_v * v_now),
        rtol * np.linalg.norm(operand))
    self.assertAllClose(w_expected, w_now, rtol=rtol, atol=atol)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_eigh_lapack_syevd(self, dtype_name="f32"):
    # For lax.linalg.eigh
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    size = 8
    operand = CompatTest.eigh_input((size, size), dtype)
    func = lambda: CompatTest.eigh_harness((8, 8), dtype)
    data = self.load_testdata(cpu_eigh_lapack_syev.data_2023_03_17[dtype_name])
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]
    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=partial(self.check_eigh_results, operand))
    # TODO(b/344892332): Remove the check after the compatibility period.
    has_xla_ffi_support = jaxlib_version >= (0, 4, 32)
    if has_xla_ffi_support:
      # FFI Kernel test
      with config.export_ignore_forward_compatibility(True):
        data = self.load_testdata(cpu_eigh_lapack_syev.data_2024_08_19[dtype_name])
        self.run_one_test(func, data, rtol=rtol, atol=atol,
                          check_results=partial(self.check_eigh_results, operand))

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}_{variant}",
           dtype_name=dtype_name, variant=variant)
      for dtype_name in ("f32", "f64")
      # We use different custom calls for sizes <= 32
      for variant in ["syevj", "syevd"])
  def test_gpu_eigh_solver_syev(self, dtype_name="f32", variant="syevj"):
    if not config.enable_x64.value and dtype_name == "f64":
      self.skipTest("Test disabled for x32 mode")
    if jtu.test_device_matches(["cuda"]):
      if _is_required_cusolver_version_satisfied(11600):
        # The underlying problem is that this test assumes the workspace size can be
        # queried from an older version of cuSOLVER and then be used in a newer one.
        self.skipTest("Newer cuSOLVER expects a larger workspace than was serialized")
      data = self.load_testdata(cuda_eigh_cusolver_syev.data_2023_03_17[f"{dtype_name}_{variant}"])
    elif jtu.test_device_matches(["rocm"]):
      data = self.load_testdata(rocm_eigh_hipsolver_syev.data_2024_08_05[f"{dtype_name}_{variant}"])
    else:
      self.skipTest("Unsupported platform")
    # For lax.linalg.eigh
    dtype = dict(f32=np.float32, f64=np.float64)[dtype_name]
    size = dict(syevj=8, syevd=36)[variant]
    rtol = dict(f32=1e-3, f64=1e-5)[dtype_name]
    atol = dict(f32=1e-2, f64=1e-10)[dtype_name]
    operand = CompatTest.eigh_input((size, size), dtype)
    func = lambda: CompatTest.eigh_harness((size, size), dtype)
    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=partial(self.check_eigh_results, operand))

  def test_tpu_Eigh(self):
    self.skipTest(
        "TODO(b/280668311): Change input matrix to not be ill-conditioned."
    )
    # For lax.linalg.eigh
    shape = (8, 8)
    dtype = np.float32
    operand = CompatTest.eigh_input(shape, dtype)
    func = lambda: CompatTest.eigh_harness(shape, dtype)
    data = self.load_testdata(tpu_Eigh.data)
    self.run_one_test(func, data, rtol=1e-3,
                      check_results=partial(self.check_eigh_results, operand))

  @staticmethod
  def lu_pivots_to_permutation_harness(shape):
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=np.int32), shape)
    return lax.linalg.lu_pivots_to_permutation(operand, permutation_size=8)

  def test_cuda_lu_pivots_to_permutation(self):
    shape = (2, 3, 4)
    func = lambda: CompatTest.lu_pivots_to_permutation_harness(shape)
    data = self.load_testdata(cuda_lu_pivots_to_permutation.data_2024_08_08)
    self.run_one_test(func, data)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}",
           dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cuda_lu_lapack_getrf(self, dtype_name:str):
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")
    if jaxlib_version < (0, 4, 32):
      self.skipTest("Not implemented in older versions of jaxlib")
    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    shape = (3, 4)
    func = lambda: CompatTest.lu_harness(shape, dtype)
    # TODO(b/360788062): Clean up after the compatibility period.
    with config.export_ignore_forward_compatibility(True):
      data = self.load_testdata(cuda_lu_cusolver_getrf.data_2024_08_19[dtype_name])
      self.run_one_test(func, data)

  @staticmethod
  def qr_harness(shape, dtype):
    # In order to keep inputs small, we construct the input programmatically
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)
    return lax.linalg.qr(operand, full_matrices=True)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_qr_lapack_geqrf(self, dtype_name="f32"):
    # For lax.linalg.qr
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    func = lambda: CompatTest.qr_harness((3, 3), dtype)
    data = self.load_testdata(cpu_qr_lapack_geqrf.data_2023_03_17[dtype_name])
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    self.run_one_test(func, data, rtol=rtol)
    # TODO(b/344892332): Remove the check after the compatibility period.
    has_xla_ffi_support = jaxlib_version >= (0, 4, 32)
    if has_xla_ffi_support:
      with config.export_ignore_forward_compatibility(True):
        # FFI Kernel test
        data = self.load_testdata(
            cpu_qr_lapack_geqrf.data_2024_08_22[dtype_name]
        )
        self.run_one_test(func, data, rtol=rtol)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}_{batched}",
           dtype_name=dtype_name, batched=batched)
      for dtype_name in ("f32",)
      # For batched qr we use cublas_geqrf_batched/hipblas_geqrf_batched.
      for batched in ("batched", "unbatched"))
  def test_gpu_qr_solver_geqrf(self, dtype_name="f32", batched="unbatched"):
    if jtu.test_device_matches(["cuda"]):
      data = self.load_testdata(cuda_qr_cusolver_geqrf.data_2023_03_18[batched])
    elif jtu.test_device_matches(["rocm"]):
      data = self.load_testdata(rocm_qr_hipsolver_geqrf.data_2024_08_05[batched])
    else:
      self.skipTest("Unsupported platform")
    # For lax.linalg.qr
    dtype = dict(f32=np.float32, f64=np.float64)[dtype_name]
    rtol = dict(f32=1e-3, f64=1e-5)[dtype_name]
    shape = dict(batched=(2, 3, 3), unbatched=(3, 3))[batched]
    func = lambda: CompatTest.qr_harness(shape, dtype)
    self.run_one_test(func, data, rtol=rtol)

  def test_tpu_Qr(self):
    # For lax.linalg.qr
    func = lambda: CompatTest.qr_harness((3, 3), np.float32)
    data = self.load_testdata(tpu_Qr.data_2023_03_17)
    self.run_one_test(func, data, rtol=1e-3, atol=1e-3)

  @staticmethod
  def lu_harness(shape, dtype):
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)
    return lax.linalg.lu(operand)

  def check_lu_results(self, operand, res_now, res_expected, *,
                       dtype, rtol=None, atol=None):
    # Same checker as in linalg_test.py
    del res_expected  # we do not check against expected
    lu_now, pivots_now, _ = res_now

    n, m = operand.shape
    self.assertEqual(n, m)
    l = np.tril(lu_now, -1) + np.eye(n, dtype=dtype)
    u = np.triu(lu_now)
    operand_copy = operand.copy()
    for i in range(n):
      operand_copy[[i, pivots_now[i]],] = operand_copy[[pivots_now[i], i],]
    self.assertAllClose(operand_copy, np.matmul(l, u), rtol=rtol, atol=atol)

  def test_tpu_Lu(self):
    # For lax.linalg.lu on TPU.
    shape = (3, 3)
    dtype = np.float32
    func = lambda: CompatTest.lu_harness(shape, dtype)
    data = self.load_testdata(tpu_Lu.data_2023_03_21)
    operand = np.reshape(np.arange(math.prod(shape), dtype=dtype), shape)
    self.run_one_test(func, data, rtol=1e-3,
                      check_results=partial(self.check_lu_results, operand,
                                            dtype=dtype))

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}",
           dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_lu_lapack_getrf(self, dtype_name:str):
    # For lax.linalg.lu on CPU.
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")
    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    shape = (3, 3)
    func = lambda: CompatTest.lu_harness(shape, dtype)
    data = self.load_testdata(cpu_lu_lapack_getrf.data_2023_06_14[dtype_name])
    operand = np.reshape(np.arange(math.prod(shape), dtype=dtype), shape)
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]
    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=partial(self.check_lu_results, operand,
                                            dtype=dtype))
    # TODO(b/344892332): Remove the check after the compatibility period.
    has_xla_ffi_support = jaxlib_version >= (0, 4, 32)
    if has_xla_ffi_support:
      with config.export_ignore_forward_compatibility(True):
        # FFI Kernel test
        data = self.load_testdata(cpu_lu_lapack_getrf.data_2024_05_31[dtype_name])
        self.run_one_test(func, data, rtol=rtol, atol=atol,
                          check_results=partial(self.check_lu_results, operand,
                                              dtype=dtype))

  def check_svd_results(self, input, res_run, res_exp,
                        rtol=None, atol=None):
    # Following linalg_test.testSVD
    def compute_max_backward_error(operand, reconstructed_operand):
      error_norm = np.linalg.norm(operand - reconstructed_operand,
                                  axis=(-2, -1))
      backward_error = (error_norm /
                        np.linalg.norm(operand, axis=(-2, -1)))
      max_backward_error = np.amax(backward_error)
      return max_backward_error

    tol = 80 * jnp.finfo(input.dtype).eps
    reconstruction_tol = 2 * tol
    unitariness_tol = tol

    out = res_run
    a = input
    compute_uv = True
    full_matrices = True
    b, m, n = input.shape
    T = lambda x: np.swapaxes(x, -1, -2)

    if compute_uv:
      # Check the reconstructed matrices
      out = list(out)
      out[1] = out[1].astype(out[0].dtype)  # for strict dtype promotion.
      if m and n:
        if full_matrices:
          k = min(m, n)
          if m < n:
            max_backward_error = compute_max_backward_error(
                a, np.matmul(out[1][..., None, :] * out[0], out[2][..., :k, :]))
            self.assertLess(max_backward_error, reconstruction_tol)
          else:
            max_backward_error = compute_max_backward_error(
                a, np.matmul(out[1][..., None, :] * out[0][..., :, :k], out[2]))
            self.assertLess(max_backward_error, reconstruction_tol)
        else:
          max_backward_error = compute_max_backward_error(
              a, np.matmul(out[1][..., None, :] * out[0], out[2]))
          self.assertLess(max_backward_error, reconstruction_tol)

      # Check the unitary properties of the singular vector matrices.
      unitary_mat = np.real(np.matmul(np.conj(T(out[0])), out[0]))
      eye_slice = np.eye(out[0].shape[-1], dtype=unitary_mat.dtype)
      self.assertAllClose(np.broadcast_to(eye_slice, (b,) + eye_slice.shape),
                          unitary_mat, rtol=unitariness_tol,
                          atol=unitariness_tol)
      if m >= n:
        unitary_mat = np.real(np.matmul(np.conj(T(out[2])), out[2]))
        eye_slice = np.eye(out[2].shape[-1], dtype=unitary_mat.dtype)
        self.assertAllClose(np.broadcast_to(eye_slice, (b,) + eye_slice.shape),
                            unitary_mat, rtol=unitariness_tol,
                            atol=unitariness_tol)
      else:
        unitary_mat = np.real(np.matmul(out[2], np.conj(np.T(out[2]))))
        eye_slice = np.eye(out[2].shape[-2], dtype=unitary_mat.dtype)
        self.assertAllClose(np.broadcast_to(eye_slice, (b,) + eye_slice.shape),
                            unitary_mat, rtol=unitariness_tol,
                            atol=unitariness_tol)
    else:
      self.assertTrue(np.allclose(np.linalg.svd(a, compute_uv=False),
                                  np.asarray(out), atol=1e-4, rtol=1e-4))

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128")])
  @jax.default_matmul_precision("float32")
  def test_cpu_schur_lapack_gees(self, dtype_name="f32"):
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    shape = (4, 4)
    input = np.arange(math.prod(shape), dtype=dtype).reshape(shape)

    def func(input):
      return lax.linalg.schur(input, compute_schur_vectors=True)

    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]

    data = self.load_testdata(cpu_schur_lapack_gees.data_2023_07_16[dtype_name])

    def check_schur_results(res_run, res_expected, *, rtol, atol):
      t_run, s_run = res_run
      self.assertAllClose(input, s_run @ t_run @ np.conj(s_run.T),
                          rtol=rtol, atol=atol)

    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=check_schur_results)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  @jax.default_matmul_precision("float32")
  def test_cpu_svd_lapack_gesdd(self, dtype_name="f32"):
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    shape = (2, 4, 4)
    input = jtu.rand_default(self.rng())(shape, dtype)
    # del input  # Input is in the testdata, here for readability
    def func(input):
      return lax.linalg.svd(input, full_matrices=True, compute_uv=True)

    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]

    data = self.load_testdata(cpu_svd_lapack_gesdd.data_2023_06_19[dtype_name])
    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=partial(self.check_svd_results,
                                            input))
    # TODO(b/344892332): Remove the check after the compatibility period.
    has_xla_ffi_support = jaxlib_version >= (0, 4, 32)
    if has_xla_ffi_support:
      with config.export_ignore_forward_compatibility(True):
        # FFI Kernel test
        data = self.load_testdata(
            cpu_svd_lapack_gesdd.data_2024_08_13[dtype_name]
        )
        self.run_one_test(func, data, rtol=rtol, atol=atol,
                          check_results=partial(self.check_svd_results, input))

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128")])
  @jax.default_matmul_precision("float32")
  def test_cpu_triangular_solve_blas_trsm(self, dtype_name="f32"):
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    a_shape = (4, 4)
    a = np.arange(math.prod(a_shape), dtype=dtype).reshape(a_shape)
    a = np.tril(a + 5 * np.eye(a.shape[-1], dtype=a.dtype))
    b_shape = (4, 5)
    b = np.arange(math.prod(b_shape), dtype=dtype).reshape(b_shape)
    left_side = True
    def func(a, b):
      return lax.linalg.triangular_solve(a, b, lower=True,
                      transpose_a=False,
                      conjugate_a=False, unit_diagonal=False,
                      left_side=left_side)

    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]

    data = self.load_testdata(cpu_triangular_solve_blas_trsm.data_2023_07_16[dtype_name])

    def check_triangular_solve_results(res_run, res_expected, *, rtol, atol):
      x, = res_run
      matmul = partial(jnp.matmul, precision=lax.Precision.HIGHEST)
      y = matmul(a, x) if left_side else matmul(x, a)
      self.assertArraysAllClose(y, jnp.broadcast_to(b, y.shape), rtol=rtol, atol=atol)

    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=check_triangular_solve_results)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  @jax.default_matmul_precision("float32")
  def test_cpu_tridiagonal_lapack_sytrd_hetrd(self, dtype_name="f32"):
    if not config.enable_x64.value and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    shape = (2, 4, 4)
    input_data = jtu.rand_default(self.rng())(shape, dtype)
    # del input_data  # Input is in the testdata, here for readability
    def func():
      return lax.linalg.tridiagonal(input_data, lower=True)

    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]

    data = self.load_testdata(
        cpu_tridiagonal_lapack_sytrd_hetrd.data_2024_09_03[dtype_name]
    )
    self.run_one_test(func, data, rtol=rtol, atol=atol)
    # TODO(b/344892332): Remove the check after the compatibility period.
    has_xla_ffi_support = jaxlib_version >= (0, 4, 32)
    if has_xla_ffi_support:
      with config.export_ignore_forward_compatibility(True):
        # FFI Kernel test
        data = self.load_testdata(
            cpu_tridiagonal_lapack_sytrd_hetrd.data_2024_09_04[dtype_name]
        )
        self.run_one_test(func, data, rtol=rtol, atol=atol)

  def test_approx_top_k(self):
    def func():
      x = np.array([3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 7.0])
      y = lax.approx_max_k(x, 3)
      z = lax.approx_max_k(x, 3)
      return y + z
    data = self.load_testdata(tpu_ApproxTopK.data_2023_05_16)
    self.run_one_test(func, data)

  def test_cuda_threefry2x32(self):
    def func(x):
      return jax.random.uniform(x, (2, 4), dtype=np.float32)

    # TODO(b/338022728): remove after 6 months
    data = self.load_testdata(cuda_threefry2x32.data_2023_03_15)
    self.run_one_test(func, data,
                      expect_current_custom_calls=["cu_threefry2x32_ffi"])

    data = self.load_testdata(cuda_threefry2x32.data_2024_07_30)
    self.run_one_test(func, data)

  def test_sharding(self):
    # Tests "Sharding", "SPMDShardToFullShape", "SPMDFullToShardShape" on TPU
    if not jtu.test_device_matches(["tpu"]) or len(jax.devices()) < 2:
      self.skipTest("Test runs only on TPU with at least 2 devices")

    # Must use exactly 2 devices for expected outputs from ppermute
    devices = jax.devices()[:2]
    mesh = Mesh(devices, axis_names=('a'))

    @partial(pjit.pjit,
             in_shardings=(P('a', None),), out_shardings=P('a', None))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('a', None),), out_specs=P('a', None))
    def func(x):  # b: f32[2, 4]
      axis_size = lax.psum(1, 'a')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(x, 'a', perm=perm)

    data = self.load_testdata(tpu_Sharding.data_2023_03_16)
    with mesh:
      self.run_one_test(func, data)

  def test_tpu_stablehlo_dynamic_reduce_window_unary(self):
    # stablehlo.dynamic_reduce_window is used temporarily on TPU for a
    # reduce window with dynamic shapes.
    # See https://github.com/openxla/stablehlo/issues/1258 for the long term.
    # The inputs are already in the test data, here only for readability.
    shape = (3, 4)
    _ = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    def func(x):
      return jnp.cumsum(x, axis=0)

    data = self.load_testdata(tpu_stablehlo_dynamic_reduce_window.data_unary_2023_06_17)
    self.run_one_test(
        func, data,
        polymorphic_shapes=("b, ...",),
        # Recent serializations also include shape_assertion, tested with dynamic_top_k
        expect_current_custom_calls=["stablehlo.dynamic_reduce_window", "shape_assertion"])

  def test_tpu_stablehlo_dynamic_reduce_window_variadic(self):
    # stablehlo.dynamic_reduce_window is used temporarily on TPU for a
    # reduce window with dynamic shapes.
    # See https://github.com/openxla/stablehlo/issues/1258 for the long term.
    # The inputs are already in the test data, here only for readability.
    shape = (3, 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    y = 100 + np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    _ = (x, y)
    def func(x, y):  # x: f32[b, 2] y: i32[b, 2]
      return lax.reduce_window(
          (x, y), (np.array(1., np.float32), np.array(2, np.int32)),
          lambda xy0, xy1: (lax.add(xy0[0], xy1[0]),
                            lax.sub(xy0[1], xy1[1])),
          (2, x.shape[0]), (1, 1), "VALID")

    data = self.load_testdata(tpu_stablehlo_dynamic_reduce_window.data_variadic_2023_06_17)
    self.run_one_test(
        func, data,
        polymorphic_shapes=("b, ...", "b, ..."),
        # Recent serializations also include shape_assertion, tested with dynamic_top_k
        expect_current_custom_calls=["stablehlo.dynamic_reduce_window", "shape_assertion"])

  @jtu.ignore_warning(
      category=FutureWarning,
      message="Raw arrays as random keys to jax.random functions are deprecated")
  def test_stablehlo_dynamic_rbg_bit_generator(self):
    # stablehlo.dynamic_rbg_bit_generator is used temporarily for a
    # rbg_bit_generator with dynamic shapes.
    # See https://github.com/openxla/stablehlo/issues/1344 for the long term.
    key = np.arange(42, 42+4, dtype=np.uint32)
    a_shape = (2, 3)
    a = np.arange(math.prod(a_shape), dtype=np.float32).reshape(a_shape)
    inputs = (key, a)
    del inputs  # already in the test data, here only for readability.

    def func(key, a):  # a is only used for its shape
      return jax.random.key_data(jax.random.split(key, a.shape[0] * a.shape[1]))

    # Note that the test currently checks that the generated sequence is the
    # same. According to the StableHLO spec: "The output is guaranteed to be
    # deterministic function of initial_state, but it is not guaranteed to be
    # deterministic between implementations"
    # See https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng_bit_generator
    # This test will fail when the implementation changes. We expect this to
    # be rare, and most users may expect the RNG sequence to be the same
    # upon reloading of a saved model.
    # In case of an intended change in behavior we will have the option to
    # replace this strict check with something else.
    data = self.load_testdata(stablehlo_dynamic_rng_bit_generator.data_2023_06_17)

    with config.default_prng_impl("unsafe_rbg"):
      self.run_one_test(
        func, data, polymorphic_shapes=(None, "b0, b1"),
        # Recent serializations also include shape_assertion, tested with dynamic_top_k
        expect_current_custom_calls=["stablehlo.dynamic_rng_bit_generator", "shape_assertion"])

  def test_stablehlo_dynamic_top_k(self):
    # stablehlo.dynamic_top_k is used temporarily for a top_k with dynamism
    a = np.arange(12, dtype=np.float32).reshape((4, 3))

    def func(a):
      return lax.top_k(a, k=a.shape[-1] - 1)

    data = self.load_testdata(stablehlo_dynamic_top_k.data_2023_07_16)
    def check_top_k_results(res_run, res_expected, *, rtol, atol):
      # The order of the results may be different, but should be the same ones
      values_expected, _ = res_expected
      values_run, indices_run = res_run
      # Check that indices are correct
      self.assertAllClose(values_run,
                          a[np.arange(a.shape[0]).reshape(a.shape[0], 1),
                            indices_run], atol=atol, rtol=rtol)
      self.assertAllClose(np.sort(values_run), np.sort(values_expected),
                          atol=atol, rtol=rtol)

    self.run_one_test(func, data,
                      polymorphic_shapes=("_, b",),
                      check_results=check_top_k_results,
                      # Recent serializations also include shape_assertion
                      expect_current_custom_calls=["stablehlo.dynamic_top_k", "shape_assertion"])

    # Now a test with serialization version 7, including shape_assertion
    data_2 = self.load_testdata(stablehlo_dynamic_top_k.data_2023_08_11)
    self.run_one_test(func, data_2,
                      polymorphic_shapes=("_, b",),
                      check_results=check_top_k_results)

  def test_dynamic_approx_top_k(self):
    # stablehlo.dynamic_approx_top_k is used temporarily for a approx_top_k
    # with dynamism
    # This is the input that was used to generate the test_data
    _ = np.arange(24, dtype=np.float32)

    def func(a):  # a: f32[b + 4]
      return lax.approx_max_k(a, k=a.shape[0] - 4)

    data = self.load_testdata(stablehlo_dynamic_approx_top_k.data_2024_05_30)

    def check_top_k_results(res_run, res_expected, *, rtol, atol):
      a = data.inputs[0]
      # The order of the results may be different, but should be the same ones
      values_expected, _ = res_expected
      values_run, indices_run = res_run
      # Check that indices are correct
      self.assertAllClose(
          values_run,
          a[indices_run],
          atol=atol,
          rtol=rtol,
      )
      self.assertAllClose(
          np.sort(values_run), np.sort(values_expected), atol=atol, rtol=rtol
      )

    self.run_one_test(
        func,
        data,
        polymorphic_shapes=("b + 4,",),
        check_results=check_top_k_results,
        expect_current_custom_calls=[
            "stablehlo.dynamic_approx_top_k",
            "shape_assertion",
        ],
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
