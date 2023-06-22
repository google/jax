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
"""Tests for backwards compatibility of custom calls.

Since we have to guarantee 6 months of backward compatibility for the
JAX serialized format, we need to guarantee that custom calls continue to
work as before. We test this here.

The tests in this file refer to the test data in ./back_compat_testdata.
There is one test for each version of a custom call target, e.g.,
`test_ducc_fft` tests the FFT custom calls on CPU.
Only custom call targets tested here should be listed in
jax_export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE. All other custom
call targets will result in an error when encountered during serialization.

Once we stop using a custom call target in JAX, you can remove it from the
_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE and you can add a comment to the
test here to remove it after 6 months.

** To create a new test **

Write the JAX function `func` that exercises the custom call `foo_call` you
want, then pick some inputs, and then add this to the new test to get started.

  def test_foo_call(self):
    def func(...): ...
    inputs = (...,)  # Tuple of nd.array, keep it small, perhaps generate the
                     # inputs in `func`.
    data = dataclasses.replace(self.load_testdata(dummy_data_dict),
                               inputs=inputs,
                               platform=self.default_jax_backend())
    self.run_one_test(func, data,
                      # Temporarily allow calls to "foo"
                      allow_additional_custom_call_targets=("foo",))

The test will fail, but will save to a file the test data you will need. The
file name will be printed in the logs. Create a new
file ./back_compat_testdata/foo_call.py and paste the test data that
you will see printed in the logs. You may want to
edit the serialization string to remove any pathnames that may be included at
the end, or gxxxxx3 at the beginning.

Name the literal `data_YYYYY_MM_DD` to include the date of serializaton
(for readability only). Then add to this file:

  from jax.experimental.jax2tf.tests.back_compat_testdata import foo_call

then update `test_custom_call_coverage`, and then update your `test_foo_call`:

  def test_foo_call(self):
    def func(...): ...
    data = self.load_testdata(foo_call.data_YYYY_MM_DD)  # <-- this is new
    self.run_one_test(func, data)

"""
import dataclasses
import datetime
from functools import partial
import itertools
import math
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from absl.testing import absltest, parameterized
from absl import logging

import numpy as np
# Import some NumPy symbols so that we can parse repr(ndarray).
from numpy import array, float32

import jax
from jax import config
from jax import core
from jax import lax
from jax import tree_util
from jax.experimental import jax2tf
from jax.experimental.jax2tf import jax_export
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_ducc_fft
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_eigh_cusolver_syev
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_eigh_lapack_syev
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_qr_cusolver_geqrf
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_qr_lapack_geqrf
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_threefry2x32
from jax.experimental.jax2tf.tests.back_compat_testdata import tf_call_tf_function
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Eigh
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Lu
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_ApproxTopK
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Qr
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Sharding
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_stablehlo_dynamic_reduce_window
from jax.experimental.jax2tf.tests.back_compat_testdata import stablehlo_dynamic_rng_bit_generator

from jax.experimental import pjit
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax._src.lib import xla_extension
from jax._src import test_util as jtu
from jax._src.interpreters import pxla
from jax._src import xla_bridge as xb

import tensorflow as tf
from tensorflow.core.framework import graph_pb2  # type: ignore[import]

config.parse_flags_with_absl()


CURRENT_TESTDATA_VERSION = 1

@dataclasses.dataclass
class CompatTestData:
  testdata_version: int
  platform: str  # One of: "cpu", "tpu", "cuda", "rocm"
  custom_call_targets: List[str]
  serialized_date: datetime.date  # e.g., datetime.date(2023, 3, 9)
  inputs: Sequence[np.ndarray]
  expected_outputs: Sequence[np.ndarray]
  mlir_module_text: str
  mlir_module_serialized: bytes
  xla_call_module_version: int  # The version of XlaCallModule to use for testing


# The dummy_data is used for getting started for adding a new test and for
# testing the helper functions.
dummy_data_dict = dict(
    testdata_version=CURRENT_TESTDATA_VERSION,
    platform="cpu",
    custom_call_targets=[],
    serialized_date=datetime.date(2023, 3, 15),
    inputs=(array(0.0, dtype=float32),),
    expected_outputs=(array(0.0, dtype=float32),),
    mlir_module_text=r"""
  module @jit_sin {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.sine %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01\x17\x05\x01\x05\x01\x03\x05\x03\x07\x07\t\x0b\x03K5\x07\x01\x1b\x07\x0b\x13\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x03\x1b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x03\x07\x0f\x17\x07\x02\xa7\x1f\x05\r\x03\x03\x03\x07\x05\x0f\x03\x0b\x0b\x1b\r'\x0f)\x031\x113\x05\x11\x05\x13\x05\x15\x05\x17\x1d\x15\x17\x05\x19\x17\x19\xef\x01\x05\x1b\x03\x03\x1d\r\x05\x1f!#%\x1d\x1d\x1d\x1f\x1d!\x1d##\x03\x03\x03+\r\x03-/\x1d%\x1d'\x1d)\x1d+)\x01\x05\x11\x03\x01\x03\x01\t\x04A\x05\x01\x11\x01\x05\x07\x03\x01\x05\x03\x11\x01\t\x05\x03\x05\x0b\x03\x01\x01\x05\x06\x13\x03\x01\x03\x01\x07\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00\x9a\x04-\x0f\x0b\x03!\x1b\x1d\x05\x1b\x83/\x1f\x15\x1d\x15\x11\x13\x15\x11\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00sine_v1\x00return_v1\x00sym_name\x00jit_sin\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(sin)/jit(main)/sin\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00\x00main\x00public\x00",
    xla_call_module_version=4,
)  # End paste


class CompatTestBase(jtu.JaxTestCase):
  """Base class with helper functions for backward compatibility tests."""
  def default_jax_backend(self) -> str:
    # Canonicalize to turn into "cuda" or "rocm"
    return xb.canonicalize_platform(jax.default_backend())

  def load_testdata(self, testdata_dict: Dict[str, Any]) -> CompatTestData:
    if testdata_dict["testdata_version"] == CURRENT_TESTDATA_VERSION:
      return CompatTestData(**testdata_dict)
    else:
      raise NotImplementedError("testdata_version not recognized: " +
                                testdata_dict["testdata_version"])

  def load_testdata_nested(self, testdata_nest) -> Iterable[CompatTestData]:
    # Load all the CompatTestData in a Python nest.
    if isinstance(testdata_nest, dict) and "testdata_version" in testdata_nest:
      yield self.load_testdata(testdata_nest)
    elif isinstance(testdata_nest, dict):
      for e in testdata_nest.values():
        yield from self.load_testdata_nested(e)
    elif isinstance(testdata_nest, list):
      for e in testdata_nest:
        yield from self.load_testdata_nested(e)
    else:
      assert False, testdata_nest

  def run_one_test(self, func: Callable[..., jax.Array],
                   data: CompatTestData,
                   polymorphic_shapes: Optional[Sequence[str]] = None,
                   rtol: Optional[float] = None,
                   atol: Optional[float] = None,
                   allow_additional_custom_call_targets: Sequence[str] = (),
                   check_results: Optional[Callable[..., None]] = None,
                   compare_with_current: bool = True):
    """Run one compatibility test.

    Args:
      func: the JAX function to serialize and run
      data: the test data
      polymorphic_shapes: when using shape polymorphism, the specification for
        each argument of `func`.
      rtol: relative tolerance for numerical comparisons
      atol: absolute tolerance for numerical comparisons
      check_results: invoked with the results obtained from running the
        serialized code, and those stored in the test data, and the kwargs rtol
        and atol.
      allow_additional_custom_call_targets: additional custom call targets to allow.
      compare_with_current: whether to compare the current behavior for
        `func` with the one stored in `data`. If `True` (default) uses the
        current version of JAX and XLA to lower and serialize `func` and check
        its results compared to the stored ones; it also dumps the current
        test data. If `False`, no current serialization are comparisons are
        done, tests only the saved serialization. Use this option for a test
        data for which we have changed the serialization.
    """
    if not isinstance(data, CompatTestData):
      raise ValueError(f"Expecting data: CompatTestData but got {data}. "
                       "Did you forget to `self.load_testdata`?")

    if self.default_jax_backend() != data.platform:
      self.skipTest(f"Test enabled only for {data.platform}")

    logging.info("Lowering and running the function at the current version")
    res_run_current = self.run_current(func, data)
    if not isinstance(res_run_current, (list, tuple)):
      res_run_current = (res_run_current,)
    res_run_current = tuple(np.array(a) for a in res_run_current)
    logging.info("Result of current version run is %s", res_run_current)

    serialized, module_str, module_version = self.serialize(
      func, data,
      polymorphic_shapes=polymorphic_shapes,
      allow_additional_custom_call_targets=allow_additional_custom_call_targets)

    custom_call_re = r"stablehlo.custom_call\s*@([^\(]+)\("
    custom_call_targets = sorted(
        list(set(re.findall(custom_call_re, module_str))))

    np.set_printoptions(threshold=sys.maxsize, floatmode="unique")
    # Print the current test data to simplify updating the test.
    updated_testdata = f"""
# Pasted from the test output (see back_compat_test.py module docstring)
data_{datetime.date.today().strftime('%Y_%m_%d')} = dict(
    testdata_version={CURRENT_TESTDATA_VERSION},
    platform={repr(self.default_jax_backend())},
    custom_call_targets={repr(custom_call_targets)},
    serialized_date={repr(datetime.date.today())},
    inputs={repr(data.inputs)},
    expected_outputs={repr(res_run_current)},
    mlir_module_text=r\"\"\"\n{module_str}\"\"\",
    mlir_module_serialized={repr(serialized)},
    xla_call_module_version={module_version},
)  # End paste

"""
    # Replace the word that should not appear.
    updated_testdata = re.sub(r"google.", "googlex", updated_testdata)
    output_dir = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR",
                           "/tmp/back_compat_testdata")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{self._testMethodName}.py")
    logging.info("Writing the updated testdata at %s", output_file)
    with open(output_file, "w") as f:
      f.write(updated_testdata)

    if rtol is None:
      rtol = 1.e-7
    if check_results is not None:
      check_results(res_run_current, data.expected_outputs, rtol=rtol,
                    atol=atol)
    else:
      self.assertAllClose(res_run_current, data.expected_outputs, rtol=rtol,
                          atol=atol)

    logging.info("Running the serialized module")
    res_run_serialized = self.run_serialized(
        data,
        polymorphic_shapes=polymorphic_shapes)
    logging.info("Result of serialized run is %s", res_run_serialized)
    if check_results is not None:
      check_results(res_run_serialized, data.expected_outputs,
                    rtol=rtol, atol=atol)
    else:
      self.assertAllClose(res_run_serialized, data.expected_outputs,
                          rtol=rtol, atol=atol)
    if compare_with_current:
      self.assertListEqual(custom_call_targets, data.custom_call_targets)

  def run_current(self, func: Callable, data: CompatTestData):
    """Lowers and runs the test function at the current JAX version."""
    return jax.jit(func)(*data.inputs)

  def serialize(self,
      func: Callable, data: CompatTestData, *,
      polymorphic_shapes: Optional[Sequence[str]] = None,
      allow_additional_custom_call_targets: Sequence[str] = ()
  ) -> Tuple[bytes, str, int]:
    """Serializes the test function.

    Args:
      func: the function to serialize
      polymorphic_shapes: the polymorphic_shapes to use for serialization
      allow_additional_custom_call_targets: whether to allow additional
        custom call targets besides the standard ones.

    Returns: a tuple with the (a) serialization, (b) the module contents as
      a string (for debugging), and (c) the module serialization version.
    """
    # Use the native exporter, to make sure we get the proper serialization.
    args_specs = jax_export.poly_specs(data.inputs, polymorphic_shapes)
    exported = jax_export.export(
      jax.jit(func),
      lowering_platform=self.default_jax_backend(),
      disabled_checks=tuple(
        jax_export.DisabledSafetyCheck.custom_call(target)
        for target in allow_additional_custom_call_targets)
    )(*args_specs)

    module_str = str(exported.mlir_module)
    serialized = exported.mlir_module_serialized
    module_version = exported.xla_call_module_version
    return serialized, module_str, module_version

  def run_serialized(self, data: CompatTestData,
                     polymorphic_shapes: Optional[Sequence[str]] = None):
    args_specs = jax_export.poly_specs(data.inputs, polymorphic_shapes)
    def ndarray_to_aval(a: np.ndarray) -> core.ShapedArray:
      return core.ShapedArray(a.shape, a.dtype)
    in_avals_tree = tree_util.tree_map(ndarray_to_aval, args_specs)
    # TODO: we ought to ensure that out_avals are polymorphic if need be. We
    # could either save the in/out_avals (but we need to first implement that
    # support in jax_export), or we can just re-use them from the current
    # exported.
    out_avals_tree = tree_util.tree_map(ndarray_to_aval, data.expected_outputs)
    # in_tree must be for (args, kwargs)
    in_avals, in_tree = tree_util.tree_flatten((in_avals_tree, {}))
    out_avals, out_tree = tree_util.tree_flatten(out_avals_tree)
    def _get_vjp(_):
      assert False  # We do not have and do not need VJP

    exported = jax_export.Exported(
        fun_name="run_serialized",
        in_tree=in_tree,
        in_avals=tuple(in_avals),
        out_tree=out_tree,
        out_avals=tuple(out_avals),
        in_shardings=(pxla.UNSPECIFIED,) * len(in_avals),
        out_shardings=(pxla.UNSPECIFIED,) * len(out_avals),
        lowering_platform=data.platform,
        disabled_checks=(),
        mlir_module_serialized=data.mlir_module_serialized,
        xla_call_module_version=data.xla_call_module_version,
        module_kept_var_idx=tuple(range(len(in_avals))),
        module_uses_dim_vars=any(not core.is_constant_shape(a.shape)
                                 for a in in_avals),
      _get_vjp=_get_vjp)

      # We use pjit in case there are shardings in the exported module.
    return pjit.pjit(jax_export.call_exported(exported))(*data.inputs)


class CompatTest(CompatTestBase):
  def test_dummy(self):
    # Tests the testing mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data, platform=self.default_jax_backend())
    self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_output(self):
    # Test the detection mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=self.default_jax_backend(),
        expected_outputs=(np.array(2.0, dtype=np.float32),))
    with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_custom_calls(self):
    # Test the detection mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=self.default_jax_backend(),
        custom_call_targets=["missing"])
    with self.assertRaisesRegex(AssertionError, "Lists differ"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_custom_call_coverage(self):
    """Tests that the back compat tests cover all the targets declared stable."""
    targets_to_cover = set(jax_export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE)
    # Add here all the testdatas that should cover the targets guaranteed
    # stable
    covering_testdatas = [
        cpu_ducc_fft.data_2023_03_17, cpu_ducc_fft.data_2023_06_14,
        cpu_eigh_lapack_syev.data_2023_03_17,
        cpu_qr_lapack_geqrf.data_2023_03_17, cuda_threefry2x32.data_2023_03_15,
        cuda_qr_cusolver_geqrf.data_2023_03_18, cuda_eigh_cusolver_syev.data_2023_03_17,
        tf_call_tf_function.data_2023_06_02,
        tpu_Eigh.data, tpu_Lu.data_2023_03_21, tpu_Qr.data_2023_03_17,
        tpu_Sharding.data_2023_03_16, tpu_ApproxTopK.data_2023_04_17,
        tpu_ApproxTopK.data_2023_05_16,
        tpu_stablehlo_dynamic_reduce_window.data_unary_2023_06_17,
        tpu_stablehlo_dynamic_reduce_window.data_variadic_2023_06_17,
        stablehlo_dynamic_rng_bit_generator.data_2023_06_17,]
    covering_testdatas = itertools.chain(
        *[self.load_testdata_nested(d) for d in covering_testdatas])
    covered_targets = set()
    for data in covering_testdatas:
      self.assertIsInstance(data, CompatTestData)
      covered_targets = covered_targets.union(data.custom_call_targets)

    covered_targets = covered_targets.union({
      # TODO(necula): add tests for eig on CPU
      "lapack_sgeev", "lapack_dgeev", "lapack_cgeev", "lapack_zgeev",
      # TODO(necula): add tests for qr on CPU in a separate change.
      "lapack_cpotrf", "lapack_dpotrf", "lapack_spotrf", "lapack_zpotrf",
      # TODO(necula): add tests for svd on CPU
      "lapack_sgesdd", "lapack_dsesdd", "lapack_cgesdd", "lapack_zgesdd",
    })
    not_covered = targets_to_cover.difference(covered_targets)
    self.assertEmpty(not_covered)

  def test_ducc_fft(self):
    def func(x):
      return lax.fft(x, fft_type="fft", fft_lengths=(4,))

    # An old lowering, with ducc_fft. We keep it for 6 months.
    data = self.load_testdata(cpu_ducc_fft.data_2023_03_17)
    # We have changed the lowering for fft, do not compare with current.
    self.run_one_test(func, data, compare_with_current=False)

    # A newer lowering, with dynamic_ducc_fft.
    data = self.load_testdata(cpu_ducc_fft.data_2023_06_14)
    self.run_one_test(func, data)

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
    if not config.jax_enable_x64 and dtype_name in ["f64", "c128"]:
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

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}_{variant}",
           dtype_name=dtype_name, variant=variant)
      for dtype_name in ("f32", "f64")
      # We use different custom calls for sizes <= 32
      for variant in ["syevj", "syevd"])
  def test_cuda_eigh_cusolver_syev(self, dtype_name="f32", variant="syevj"):
    # For lax.linalg.eigh
    dtype = dict(f32=np.float32, f64=np.float64)[dtype_name]
    size = dict(syevj=8, syevd=36)[variant]
    rtol = dict(f32=1e-3, f64=1e-5)[dtype_name]
    atol = dict(f32=1e-2, f64=1e-10)[dtype_name]
    operand = CompatTest.eigh_input((size, size), dtype)
    func = lambda: CompatTest.eigh_harness((size, size), dtype)
    data = self.load_testdata(cuda_eigh_cusolver_syev.data_2023_03_17[f"{dtype_name}_{variant}"])
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
  def qr_harness(shape, dtype):
    # In order to keep inputs small, we construct the input programmatically
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)
    return lax.linalg.qr(operand, full_matrices=True)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_qr_lapack_geqrf(self, dtype_name="f32"):
    # For lax.linalg.qr
    if not config.jax_enable_x64 and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    func = lambda: CompatTest.qr_harness((3, 3), dtype)
    data = self.load_testdata(cpu_qr_lapack_geqrf.data_2023_03_17[dtype_name])
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    self.run_one_test(func, data, rtol=rtol)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}_{batched}",
           dtype_name=dtype_name, batched=batched)
      for dtype_name in ("f32",)
      # For batched qr we use cublas_geqrf_batched
      for batched in ("batched", "unbatched"))
  def test_cuda_qr_cusolver_geqrf(self, dtype_name="f32", batched="unbatched"):
    # For lax.linalg.qr
    dtype = dict(f32=np.float32, f64=np.float64)[dtype_name]
    rtol = dict(f32=1e-3, f64=1e-5)[dtype_name]
    shape = dict(batched=(2, 3, 3), unbatched=(3, 3))[batched]
    func = lambda: CompatTest.qr_harness(shape, dtype)
    data = self.load_testdata(cuda_qr_cusolver_geqrf.data_2023_03_18[batched])
    self.run_one_test(func, data, rtol=rtol)

  def test_tpu_Qr(self):
    # For lax.linalg.qr
    func = lambda: CompatTest.qr_harness((3, 3), np.float32)
    data = self.load_testdata(tpu_Qr.data_2023_03_17)
    self.run_one_test(func, data, rtol=1e-3)

  @staticmethod
  def lu_harness(shape, dtype):
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)
    return lax.linalg.lu(operand)

  def test_tpu_Lu(self):
    # For lax.linalg.lu
    func = lambda: CompatTest.lu_harness((3, 3), np.float32)
    data = self.load_testdata(tpu_Lu.data_2023_03_21)
    self.run_one_test(func, data, rtol=1e-3)

  def test_approx_top_k(self):
    def func():
      x = np.array([3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 7.0])
      y = lax.approx_max_k(x, 3)
      z = lax.approx_max_k(x, 3)
      return y + z
    data = self.load_testdata(tpu_ApproxTopK.data_2023_05_16)
    self.run_one_test(func, data)

  def test_cu_threefry2x32(self):
    def func(x):
      return jax.random.uniform(x, (2, 4), dtype=np.float32)

    data = self.load_testdata(cuda_threefry2x32.data_2023_03_15)
    self.run_one_test(func, data)

  def test_sharding(self):
    # Tests "Sharding", "SPMDShardToFullShape", "SPMDFullToShardShape" on TPU
    if jtu.device_under_test() != "tpu" or len(jax.devices()) < 2:
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
        polymorphic_shapes=("b, ...",))

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
        polymorphic_shapes=("b, ...", "b, ..."))

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

    prev_default_prng_impl = jax.config.jax_default_prng_impl
    try:
      jax.config.update("jax_default_prng_impl", "unsafe_rbg")

      self.run_one_test(func, data, polymorphic_shapes=(None, "b0, b1"))
    finally:
      jax.config.update("jax_default_prng_impl", prev_default_prng_impl)


class CompatTensoflowTest(CompatTestBase):
  """Compatibility tests that use TF.

  Uses tf.Graph to serialize and run the functions; expects that `func`
  contains a `jax2tf.call_tf` and uses `jax2tf.convert` to generate a
  `tf.Graph` containing a XlaCallModule with the actual MLIR module.
  """

  def run_current(self, func: Callable, data: CompatTestData):
    # Is there a better way to serialize/deserialize TF functions? I thought
    # about using tf.saved_model, but then we have to zip/unzip a whole
    # directory.
    @tf.function(autograph=False, jit_compile=True)
    def tf_func(the_input):  # Use recognizeable names for input and result
      res = jax2tf.convert(func, native_serialization=True)(the_input)
      return tf.identity(res, name="the_result")

    self.tf_func = tf_func
    return tf_func(*data.inputs)  # type: ignore

  def serialize(self, func: Callable, data: CompatTestData,
                polymorphic_shapes: Optional[Sequence[str]] = None,
                allow_additional_custom_call_targets: Sequence[str] = ()):
    # We serialize as a tf.Graph
    assert len(data.inputs) == 1  # We only support a single input now
    tf_graph = self.tf_func.get_concrete_function(*data.inputs).graph
    for op in tf_graph.get_operations():
      if op.type == "XlaCallModule":
        serialized_module = op.get_attr("module")
        module_str = xla_extension.mlir.deserialize_portable_artifact(
          serialized_module)
        module_version = op.get_attr("version")
        break
    else:
      raise ValueError("Cannot find an XlaCallModule")
    tf_graph_def = tf_graph.as_graph_def()
    # module_str is just for human readability, add both the MLIR module
    # and the tf.Graph
    module_str = ("# First the MLIR module:\n" + module_str +
                  "\n# Then the tf.Graph:\n" + str(tf_graph_def))
    serialized = tf_graph_def.SerializeToString()
    return serialized, module_str, module_version

  def run_serialized(self, data: CompatTestData,
                     polymorphic_shapes: Optional[Sequence[str]] = None):
    loaded_f_tf_graph = graph_pb2.GraphDef()
    loaded_f_tf_graph.ParseFromString(data.mlir_module_serialized)

    @tf.function(autograph=False)
    def loaded_fun(x):
      result = tf.import_graph_def(loaded_f_tf_graph,
                                   input_map={"the_input": x},
                                   return_elements=["the_result:0"])
      return result[0]

    return (loaded_fun(*data.inputs).numpy(),)

  def test_tf_call_tf_function(self):
    self.skipTest("b/286409830: brittle on function naming.")
    # A custom call tf.call_tf_function is generated when we lower call_tf
    # with the call_tf_graph=True option.
    def func(x):
      def func_tf(x):
        return tf.math.sin(x)
      return jnp.cos(jax2tf.call_tf(func_tf, output_shape_dtype=x,
                                    call_tf_graph=True)(x))

    data = self.load_testdata(tf_call_tf_function.data_2023_06_02)
    self.run_one_test(func, data)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
