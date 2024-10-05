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

from __future__ import annotations

import contextlib
from collections.abc import Callable
from functools import partial
import itertools
import logging
import os
import re
import time
import unittest
from unittest import SkipTest

from absl.testing import absltest

import jax
from jax import ad_checkpoint
from jax import dtypes
from jax import lax
from jax import numpy as jnp
from jax.experimental import host_callback as hcb
from jax._src import core
from jax._src import xla_bridge
from jax._src import test_util as jtu
from jax._src.lib import xla_client

from jax.experimental.host_callback import _deprecated_id_print as hcb_id_print

xops = xla_client.ops

import numpy as np

jax.config.parse_flags_with_absl()


class _TestingOutputStream:
  """Use as `output_stream` for tests."""

  def __init__(self):
    self._output = []
    self._test_method_name = None

  def write(self, what: str) -> None:
    logging.info(f"output_stream[{self._test_method_name}]: {what}")
    self._output.append(what)

  @property
  def output(self):
    return "".join(self._output)

  @property
  def output_sorted_by_device(self):
    # Assume that the output is a sequence of strings including metadata
    # and data, with metadata containing `device: xxx`
    by_device = []  # each element is a pair (device, str_list)
    for s in self._output:
      m = re.match(r".*device: (\S+)", s)
      if m:
        by_device.append((m.group(1), []))
      assert by_device, f"output does not include 'device:': {self._output}"
      by_device[-1][1].append(s)

    sorted_by_device = sorted(by_device, key=lambda x: x[0])
    return "\n".join(itertools.chain(*[s[1] for s in sorted_by_device]))

  def __str__(self):
    return "TestingOutputStream"

  def reset(self):
    self._output = []


testing_stream = _TestingOutputStream()


def fun1(a):
  """Function used for several `id_tap` tests."""
  y = hcb_id_print(a * 2., what="a * 2", output_stream=testing_stream,
                   callback_flavor=hcb.CallbackFlavor.DEBUG)
  y = hcb_id_print(y * 3., what="y * 3", output_stream=testing_stream, result=y,
                   callback_flavor=hcb.CallbackFlavor.DEBUG)
  return y ** 2  # Some computation to make the gradient interesting


def fun1_equiv(a):  # Numerical equivalent of fun1
  return (a * 2.) ** 2


def maybe_print(do_print: bool,
                arg,
                what: str,
                tap_with_device: bool | None = False,
                device_index: int = 0):
  """Conditionally print on testing_string"""
  if do_print:
    return hcb_id_print(
        arg,
        what=what,
        output_stream=testing_stream,
        tap_with_device=tap_with_device,
        device_index=device_index)
  else:
    return arg


def local_devices():
  # Tests require using not more than 2 devices.
  return jax.local_devices()[:2]


ignore_jit_of_pmap_warning = partial(
    jtu.ignore_warning, message=".*jit-of-pmap.*")


def assertMultiLineStrippedEqual(tst: jtu.JaxTestCase,
                                 expected: str, what: str):
  """A variant that preprocesses the string to eliminate non-determinism in
  floating point values, and several uninteresting id_tap primitive params.
  """

  # Sometimes we get floating points in the output; we round them
  def repl_floats(match_group):
    matched = match_group.group(0)
    if matched == ".": return matched
    x = np.around(float(matched), decimals=2)
    return f"{x:.2f}"

  what = re.sub(r"\-?\d+\.[\-\def]*", repl_floats, what)
  what = re.sub(r"output_stream=[^\]\n,]*,?", "", what)
  what = re.sub(r"threshold=[^\]\n,]*,?", "", what)
  what = re.sub(r"bwd=[^\]\n]*", "", what)
  what = re.sub(r"out_trees=[^\]\n]*", "", what)
  what = re.sub(r"fwd_jaxpr_thunk=[^\]\n]*", "", what)
  what = re.sub(r"jvp_jaxpr_thunk=[^\]\n]*", "", what)
  # Empty lines
  what = re.sub(r"^\s*\n", "", what, flags=re.MULTILINE)

  def repl_func(match_group):
    matched = match_group.group(3)
    if "function _print_consumer" in matched:
      return match_group.group(1) + "=_print"
    else:
      return match_group.group(1) + "=..."

  what = re.sub(r"((tap_func_)|(callback))=([^\]\n,]*),?", repl_func, what)
  tst.assertMultiLineStrippedEqual(expected, what)


def helper_set_hlo_dump():
  flags_str = os.getenv("XLA_FLAGS", "")
  import shutil
  dump_dir = "/tmp/xla_dump"
  os.environ["XLA_FLAGS"] = f"{flags_str} --xla_dump_to={dump_dir}"
  if os.path.isdir(dump_dir):
    logging.warning("Deleting old XLA dump directory %s", dump_dir)
    shutil.rmtree(dump_dir)
  logging.warning("Setting XLA dump directory %s", dump_dir)
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()


def helper_print_optimized_hlo(fun, *args):
  backend = xla_bridge.get_backend(platform=jtu.device_under_test())
  c = jax.jit(fun, backend=backend.platform).lower(*args)
  logging.info(re.sub(r", metadata.*", "", c.compile().as_text()))


def helper_log_ir(name,
                  f_jax,
                  *args,
                  num_partitions=None,
                  strip_metadata=False):
  logging.info(f"Jaxpr[{name}]: {jax.make_jaxpr(f_jax)(*args)}")
  jax_comp = f_jax.lower(*args)
  logging.info(f"HLO[{name}]: {jax_comp.compiler_ir(dialect='hlo').as_hlo_text()}")
  jax_optimized_hlo = jax_comp.compile().as_text()
  if strip_metadata:
    jax_optimized_hlo = re.sub(r", metadata.*", "", jax_optimized_hlo)
  logging.info(f"Optimized HLO[{name}]: {jax_optimized_hlo}")


_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(2))

def tearDownModule():
  _exit_stack.close()


def assertMultiDeviceOutputEqual(tst: jtu.JaxTestCase,
                                 expected_2CPUs: str):
  """Check that the multi-device output is equal to the expected.

  The tests run with 2 devices if available, otherwise 1 device.
  We adjust the expected output here for 1 device.

  Args:
    expected_2CPUs: the expected output for 2 CPUs. If there is only
      one device, this is trimmed to the first device. If the current
      device_under_test is not a CPU, then we change the names
  """
  expected = expected_2CPUs
  if len(local_devices()) == 1:
    start_device_1 = expected.find('device: cpu:1')
    if start_device_1 >= 0:
      expected = expected[0:start_device_1]

  def replace_device_name(m) -> str:
    return str(local_devices()[int(m.group(1))])

  expected = re.sub(r'cpu:(\d+)', replace_device_name, expected)
  what = testing_stream.output_sorted_by_device
  return assertMultiLineStrippedEqual(tst, expected, what)


class HostCallbackImportsTest(jtu.JaxTestCase):
  @jtu.ignore_warning(
      category=DeprecationWarning,
      message="The host_callback APIs are deprecated")
  def test_deprecated_imports(self):
    if hasattr(hcb, "id_print"):
      id_print = hcb.id_print
      self.assertIs(id_print, hcb_id_print)

class HostCallbackTapTest(jtu.JaxTestCase):

  def setUp(self):
    # skipping here skips teardown, so do this before super().setUp().
    if jtu.test_device_matches(["gpu"]) and jax.device_count() > 1:
      raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")
    if xla_bridge.using_pjrt_c_api():
      raise SkipTest("host_callback not implemented in PJRT C API")
    super().setUp()
    self.enter_context(jtu.ignore_warning(
      category=DeprecationWarning, message="The host_callback APIs are deprecated"))
    self.enter_context(jtu.ignore_warning(
      category=DeprecationWarning, message="backend and device argument"))
    testing_stream.reset()
    testing_stream._test_method_name = self._testMethodName
    self.old_flags = os.getenv("XLA_FLAGS", "")

  def tearDown(self) -> None:
    if os.getenv("XLA_FLAGS") != self.old_flags:
      os.environ["XLA_FLAGS"] = self.old_flags
      xla_bridge.get_backend.cache_clear()
    jax.effects_barrier()
    super().tearDown()

  def test_tap_eval(self):
    self.assertAllClose((5. * 2.) ** 2, fun1(5.))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        what: a * 2
        10.00
        what: y * 3
        30.00""", testing_stream.output)

  def test_tap_with_tuple_results(self):
    def func2(x):
      x1, y1 = hcb_id_print((x * 2., x * 3.), output_stream=testing_stream)
      return x1 + y1

    self.assertEqual(3. * (2. + 3.), func2(3.))
    jax.effects_barrier()

    assertMultiLineStrippedEqual(self, """
        ( 6.00 9.00 )""", testing_stream.output)

  def test_tap_with_dict_results(self):
    def func2(x):
      res = hcb_id_print(dict(a=x * 2., b=x * 3.), output_stream=testing_stream)
      return res["a"] + res["b"]

    self.assertEqual(3. * (2. + 3.), func2(3.))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        { a=6.00 b=9.00 }""", testing_stream.output)

  def test_tap_with_result(self):
    def func2(x):
      x1 = hcb_id_print((x * 2., x * 3.), result=x * 4.,
                        output_stream=testing_stream)
      return x1

    self.assertEqual(3. * 4., func2(3.))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        ( 6.00 9.00 )""", testing_stream.output)

  def test_tap_empty(self):
    """Tap empty arrays."""
    hcb_id_print((), output_stream=testing_stream)
    hcb_id_print((1., np.ones((2, 0))), what="second", output_stream=testing_stream)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        (  )
        what: second
        ( 1.00 [] )""", testing_stream.output)

  def test_tap_jit_simple(self):
    jit_fun1 = jax.jit(lambda x: 3. * hcb_id_print(
        2. * x, what="here", output_stream=testing_stream))
    self.assertAllClose(6. * 5., jit_fun1(5.))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        what: here
        10.00""", testing_stream.output)

  def test_tap_jit_no_invars(self):
    def func():  # jitted function does not take arguments
      return hcb_id_print(42, output_stream=testing_stream)

    self.assertAllClose(42, jax.jit(func)())
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
    42""", testing_stream.output)

  def test_tap_jit_multiple_invars(self):
    def func(x1, x2):
      return hcb_id_print(x1 + x2, output_stream=testing_stream)

    self.assertAllClose(42, jax.jit(func)(40, 2))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
    42""", testing_stream.output)

  def test_tap_jit_constant(self):
    def func(x):
      return hcb_id_print(42, result=x, output_stream=testing_stream)

    self.assertAllClose(5, jax.jit(func)(5))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
    42""", testing_stream.output)

  def test_tap_jit_sequence1(self):
    def func(x):
      x1 = hcb_id_print(x, where="1", output_stream=testing_stream)
      return hcb_id_print(x1 + 1, where="2", output_stream=testing_stream)

    logging.info("%s: %s", self._testMethodName,
                 jax.make_jaxpr(func)(1))
    logging.info(
        "%s: %s",
        self._testMethodName,
        jax.jit(func)
        .trace(1)
        .lower(lowering_platforms=(jtu.device_under_test(),)).as_text("hlo"))
    self.assertEqual(2, jax.jit(func)(1))
    jax.effects_barrier()

    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: 2
        2""", testing_stream.output)

  def test_tap_jit2(self):
    """A sequence of JIT."""

    def func(x):
      x1 = hcb_id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb_id_print(x1 + 1, where="2", output_stream=testing_stream)
      return x2

    self.assertEqual(2, jax.jit(func)(1))
    self.assertEqual(11, jax.jit(func)(10))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: 2
        2
        where: 1
        10
        where: 2
        11""", testing_stream.output)

  def test_tap_jit_result_unused(self):
    """We can id_print even if we don't use the result."""

    def func(x):
      hcb_id_print(x, where="1", output_stream=testing_stream)
      hcb_id_print(x + 1, where="2", output_stream=testing_stream)
      return x + 1

    self.assertEqual(2, jax.jit(func)(1))
    self.assertEqual(11, jax.jit(func)(10))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: 2
        2
        where: 1
        10
        where: 2
        11""", testing_stream.output)

  def test_tap_jit_nested(self):
    def func(x):
      x1 = hcb_id_print(x, where="1", output_stream=testing_stream)

      def func_nested(x):
        x2 = hcb_id_print(x + 1, where="nested", output_stream=testing_stream)
        return x2

      x3 = jax.jit(func_nested)(x1)
      return hcb_id_print(x3 + 1, where="3", output_stream=testing_stream)

    self.assertEqual(3, jax.jit(func)(1))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: nested
        2
        where: 3
        3""", testing_stream.output)

  @jtu.sample_product(with_jit=[True, False])
  def test_tap_cond(self, with_jit=False):
    """A conditional"""

    def func(x):
      x1 = hcb_id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb_id_print(x1 + 1, where="2", output_stream=testing_stream)

      x4 = lax.cond(x % 2 == 0,
                    lambda x: hcb_id_print(x, where="cond_t",
                                           output_stream=testing_stream),
                    lambda x: hcb_id_print(-1, where="cond_f", result=x,
                                           output_stream=testing_stream),
                    x2 + 1)
      x5 = hcb_id_print(x4 + 1, where="end", output_stream=testing_stream)
      return x5

    transform = jax.jit if with_jit else lambda f: f
    self.assertEqual(4, transform(func)(1))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: 2
        2
        where: cond_f
        -1
        where: end
        4""", testing_stream.output)

  @jtu.sample_product(with_jit=[True, False])
  def test_tap_while_cond(self, with_jit=False):
    def func(x):
      x1 = hcb_id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb_id_print(x1 + 1, where="2", output_stream=testing_stream)

      def body(x):
        x3 = hcb_id_print(x, where="w_b_1", output_stream=testing_stream)
        x4 = lax.cond(x % 2 == 0,
                      lambda x: hcb_id_print(x, where="w_b_t",
                                             output_stream=testing_stream),
                      lambda x: hcb_id_print(-1, where="w_b_f",
                                             result=x, output_stream=testing_stream),
                      x3 + 1)
        return hcb_id_print(x4, where="w_b_2", output_stream=testing_stream)

      x10 = lax.while_loop(lambda x: x <= 3, body, x2)
      res = hcb_id_print(x10, where="end", output_stream=testing_stream)
      return res

    transform = jax.jit if with_jit else lambda f: f
    self.assertEqual(4, transform(func)(1))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: 2
        2
        where: w_b_1
        2
        where: w_b_t
        3
        where: w_b_2
        3
        where: w_b_1
        3
        where: w_b_f
        -1
        where: w_b_2
        4
        where: end
        4""", testing_stream.output)

  def test_tap_jit_while_pred_tap(self):
    """While with printing in the conditional."""

    def func(x):
      x1 = hcb_id_print(x, where="1")
      x10 = lax.while_loop(lambda x: hcb_id_print(x < 3,
                                                  where="w_p",
                                                  output_stream=testing_stream),
                           lambda x: hcb_id_print(x + 1, where="w_b",
                                                  output_stream=testing_stream),
                           x1)
      res = hcb_id_print(x10, where="3", output_stream=testing_stream)
      return res

    self.assertEqual(3, jax.jit(func)(1))
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self,
                                 """
                                 where: w_p
                                 True
                                 where: w_b
                                 2
                                 where: w_p
                                 True
                                 where: w_b
                                 3
                                 where: w_p
                                 False
                                 where: 3
                                 3""", testing_stream.output)

  @jtu.sample_product(with_jit=[True, False])
  def test_tap_scan_cond(self, with_jit=True):
    def func(x):
      x1 = hcb_id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb_id_print(x1 + 1, where="2", output_stream=testing_stream)

      def body(c, x):
        x3 = hcb_id_print(x, where="s_1", output_stream=testing_stream)
        x4 = lax.cond(x % 2 == 0,
                      lambda x: hcb_id_print(x, where="s_t", output_stream=testing_stream),
                      lambda x: hcb_id_print(-1, where="s_f", result=x, output_stream=testing_stream),
                      x3 + 1)
        return (c, hcb_id_print(x4, where="s_2", output_stream=testing_stream))

      _, x10 = lax.scan(body, x2, jnp.arange(3))
      res = hcb_id_print(x10, where="10", output_stream=testing_stream)
      return res

    if with_jit:
      func = jax.jit(func)
    res = func(1)
    self.assertAllClose(jnp.arange(1, 4), res)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: 2
        2
        where: s_1
        0
        where: s_t
        1
        where: s_2
        1
        where: s_1
        1
        where: s_f
        -1
        where: s_2
        2
        where: s_1
        2
        where: s_t
        3
        where: s_2
        3
        where: 10
        [1 2 3]""", testing_stream.output)
    testing_stream.reset()

  @jtu.sample_product(
    nr_args=[1, 2],
    shape=[(), (2,), (2, 3), (2, 3, 4)],
    dtype=jtu.dtypes.all,
  )
  def test_tap_jit_dtypes(self, nr_args=2, dtype=jnp.int16, shape=(2,)):
    if dtype in (jnp.complex64, jnp.complex128, jnp.bool_):
      raise SkipTest(f"host_callback not implemented for {dtype}.")
    if dtype == np.bool_:
      args = [self.rng().choice(a=[True, False], size=shape)]
    else:
      args = [jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)]
    if nr_args > 1:
      args = args * nr_args
    jit_fun1 = jax.jit(lambda xs: hcb_id_print(
        xs,
        a_new_test="************",
        testcase_name=f"{shape=}_{dtype=}_{nr_args=}"))

    res = jit_fun1(args)
    self.assertAllClose(args, res, check_dtypes=True)

  def test_tap_jit_large(self):
    arg = jnp.arange(10000, dtype=jnp.int32).reshape((10, 10, 5, -1))
    jax.jit(hcb_id_print)(arg)

  def test_tap_jit_several_together(self):
    arg = jnp.arange(50, dtype=jnp.int32).reshape((10, 5))
    jax.jit(lambda x, y: hcb_id_print((x, y, x * 2)))(arg, jnp.ones(100, dtype=jnp.int32))

  def test_tap_while(self):
    """Executing while, even without JIT uses compiled code"""
    y = jnp.ones(5)  # captured const

    def func(x):
      return lax.while_loop(
          lambda c: c[1] < 5,
          lambda c: (y, hcb_id_print(c[1], output_stream=testing_stream) + 1),
          (x, 1))

    func(y)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        1
        2
        3
        4""", testing_stream.output)

  def test_tap_grad_primal_unused(self):
    # The output of id_print is not needed for backwards pass
    def func(x):
      return 2. * hcb_id_print(x * 3., what="x * 3",
                               output_stream=testing_stream,
                               callback_flavor=hcb.CallbackFlavor.DEBUG)

    grad_func = jax.grad(func)
    arg = jnp.float32(5.)
    jaxpr = str(jax.make_jaxpr(grad_func)(arg))
    # making the Jaxpr does not print anything
    jax.effects_barrier()

    if hcb._HOST_CALLBACK_LEGACY.value:
      treedef = jax.tree.structure(arg)
      assertMultiLineStrippedEqual(
          self, f"""
        {{ lambda ; a:f32[]. let
            b:f32[] = mul a 3.00
            c:f32[] = outside_call[
              arg_treedef={treedef}
              callback=...
              device_index=0
              identity=True
            ] b
            _:f32[] = mul 2.00 c
            d:f32[] = mul 2.00 1.00
            e:f32[] = mul d 3.00
          in (e,) }}""", jaxpr)
    assertMultiLineStrippedEqual(self, "", testing_stream.output)
    testing_stream.reset()

    res_grad = grad_func(arg)
    jax.effects_barrier()

    self.assertAllClose(6., res_grad, check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
        what: x * 3
        15.00""", testing_stream.output)

  def test_tap_grad_simple(self):
    def func(x):
      y = hcb_id_print(x * 2., what="x * 2", output_stream=testing_stream,
                       callback_flavor=hcb.CallbackFlavor.DEBUG)
      return x * hcb_id_print(y * 3., what="y * 3",
                              output_stream=testing_stream,
                              callback_flavor=hcb.CallbackFlavor.DEBUG)

    grad_func = jax.grad(func)

    res_grad = grad_func(jnp.float32(5.))
    self.assertAllClose(2. * 5. * 6., res_grad, check_dtypes=False)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        what: x * 2
        10.00
        what: y * 3
        30.00""", testing_stream.output)

  def test_tap_grad_grad(self):
    def func(x):
      y = hcb_id_print(x * 2., what="x * 2", output_stream=testing_stream,
                       callback_flavor=hcb.CallbackFlavor.DEBUG)
      return x * (y * 3.)

    grad_func = jax.grad(jax.grad(func))
    # making the Jaxpr does not print anything
    _ = jax.make_jaxpr(grad_func)(5.)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, "", testing_stream.output)

    res_grad = grad_func(jnp.float32(5.))

    self.assertAllClose(12., res_grad, check_dtypes=False)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        what: x * 2
        10.00""", testing_stream.output)

  def test_tap_grad_pytree(self):
    def func(x):
      x4, x5 = hcb_id_print((x * 2., x * 3.), what="pair",
                            result=(x * 4., x * 5.),
                            output_stream=testing_stream,
                            callback_flavor=hcb.CallbackFlavor.DEBUG)
      return x4 + 2. * x5

    x = jnp.float32(5.)
    grad_func = jax.grad(func)
    print(jax.make_jaxpr(grad_func)(x))
    res_grad = grad_func(x)
    self.assertAllClose(14., res_grad, check_dtypes=False)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        what: pair
        ( 10.00 15.00 )""", testing_stream.output)

  def test_tap_grad_float0(self):

    def func(x, yint):
      x, yint = hcb_id_print((x, yint), what="pair", output_stream=testing_stream,
                             callback_flavor=hcb.CallbackFlavor.DEBUG)
      return x * yint.astype(x.dtype)

    grad_func = jax.grad(func)

    res_grad = grad_func(jnp.float32(5.), jnp.int32(2))
    self.assertAllClose(2., res_grad, check_dtypes=False)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        what: pair
        ( 5.00 2 )""", testing_stream.output)

  def test_tap_grad_float0_result(self):
    # https://github.com/jax-ml/jax/issues/7340
    # x is a Tuple[f32[2], s32[3]]
    x = (np.array([.7, .8], dtype=np.float32),
         np.array([11, 12, 13], dtype=np.int32))
    def f_jax(x):
      x = hcb_id_print(x, result=x, output_stream=testing_stream,
                       callback_flavor=hcb.CallbackFlavor.DEBUG)  # result= is important
      return (3. * x[0], x[1])

    def f_jax_vjp(x):
      res, pullback = jax.vjp(f_jax, x)
      g, = pullback((np.ones(x[0].shape, dtype=x[0].dtype),
                     np.zeros(x[1].shape, dtype=dtypes.float0)))
      return g

    g = f_jax_vjp(x)
    self.assertAllClose(np.array([3., 3.], dtype=np.float32), g[0])
    self.assertEqual(dtypes.float0, g[1].dtype)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        ( [0.70 0.80] [11 12 13] )""", testing_stream.output)

  def test_tap_higher_order_grad_float0_result(self):
    # https://github.com/jax-ml/jax/issues/7340
    # x is a Tuple[f32[2], s32[3]]
    x = (np.array([.7, .8], dtype=np.float32),
         np.array([11, 12, 13], dtype=np.int32))
    def f_jax(x):
      x = hcb_id_print(x, result=x, output_stream=testing_stream,
                       callback_flavor=hcb.CallbackFlavor.DEBUG)  # result= is important
      return (jnp.sin(x[0]), x[1])

    def wrap_vjp(f, args, res_f_of_args):
      # Given a function "f" and "args" return the f_vjp and args_vjp
      def make_ct(res):
        res_dtype = np.result_type(res)
        if res_dtype == dtypes.float0:
          return res
        ct_dtype = core.primal_dtype_to_tangent_dtype(res_dtype)
        return np.ones(np.shape(res), dtype=ct_dtype)
      cts = jax.tree.map(make_ct, res_f_of_args)
      def f_vjp(args, cts):
        res, pullback = jax.vjp(f, *args)
        return pullback(cts)
      return (f_vjp, (args, cts))

    res = f_jax(x)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        ( [0.70 0.80] [11 12 13] )""", testing_stream.output)
    testing_stream.reset()

    # 1st order
    f_jax_vjp1, args_vjp1 = wrap_vjp(f_jax, (x,), res)
    res_vjp1 = f_jax_vjp1(*args_vjp1)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        ( [0.70 0.80] [11 12 13] )""", testing_stream.output)
    testing_stream.reset()

    # 2nd order
    f_jax_vjp2, args_vjp2 = wrap_vjp(f_jax_vjp1, args_vjp1, res_vjp1)
    res_vjp2 = f_jax_vjp2(*args_vjp2)

    # 3rd order
    f_jax_vjp3, args_vjp3 = wrap_vjp(f_jax_vjp2, args_vjp2, res_vjp2)
    _ = f_jax_vjp3(*args_vjp3)

  def test_tap_vmap(self):
    vmap_fun1 = jax.vmap(fun1)
    vargs = jnp.array([jnp.float32(4.), jnp.float32(5.)])
    vmap_fun1(vargs)
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      assertMultiLineStrippedEqual(self, """
          transforms: [('batch', {'batch_dims': (0,)})] what: a * 2
          [ 8.00 10.00]
          transforms: [('batch', {'batch_dims': (0,)})] what: y * 3
          [24.00 30.00]""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          what: a * 2
          8.00
          what: a * 2
          10.00
          what: y * 3
          24.00
          what: y * 3
          30.00
      """, testing_stream.output)

  def test_tap_vmap_not_batched(self):
    x = 3.

    def func(y):
      # x is not mapped, y is mapped
      _, y = hcb_id_print((x, y), output_stream=testing_stream,
                          callback_flavor=hcb.CallbackFlavor.DEBUG)
      return x + y

    vmap_func = jax.vmap(func)
    vargs = jnp.array([jnp.float32(4.), jnp.float32(5.)])
    _ = vmap_func(vargs)
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      assertMultiLineStrippedEqual(self, """
        transforms: [('batch', {'batch_dims': (None, 0)})]
        ( 3.00 [4.00 5.00] )""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
        ( 3.00 4.00 )
        ( 3.00 5.00 )
        """, testing_stream.output)

  def test_tap_vmap_vmap(self):
    # A 2D tensor with x[i, j] = i + j using 2 vmap
    def sum(x, y):
      return hcb_id_print(x + y, output_stream=testing_stream,
                          callback_flavor=hcb.CallbackFlavor.DEBUG)

    def sum_rows(xv, y):
      return jax.vmap(sum, in_axes=(0, None))(xv, y)

    def sum_all(xv, yv):
      return jax.vmap(sum_rows, in_axes=(None, 0))(xv, yv)

    xv = jnp.arange(5, dtype=np.int32)
    yv = jnp.arange(3, dtype=np.int32)
    # assertMultiLineStrippedEqual(self, "", str(jax.make_jaxpr(sum_all)(xv, yv)))
    _ = sum_all(xv, yv)
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      assertMultiLineStrippedEqual(self, """
          transforms: [('batch', {'batch_dims': (0,)}), ('batch', {'batch_dims': (0,)})]
          [[0 1 2 3 4]
          [1 2 3 4 5]
          [2 3 4 5 6]]""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          0
          1
          2
          1
          2
          3
          2
          3
          4
          3
          4
          5
          4
          5
          6
      """, testing_stream.output)

  def test_tap_vmap_while(self):
    """Vmap of while."""

    def func(x):
      # like max(x, 2)
      x1 = hcb_id_print(x, where="before:x", output_stream=testing_stream,
                        callback_flavor=hcb.CallbackFlavor.DEBUG)
      x2 = lax.while_loop(
          lambda x: x < 2, lambda x: hcb_id_print(
              x + 1, where="body:x+1", output_stream=testing_stream,
              callback_flavor=hcb.CallbackFlavor.DEBUG), x1)
      res = hcb_id_print(x2, where="after:x", output_stream=testing_stream,
                         callback_flavor=hcb.CallbackFlavor.DEBUG)
      return res

    inputs = np.arange(5, dtype=np.int32)
    self.assertAllClose(
        np.array([2, 2, 2, 3, 4]),
        jax.jit(jax.vmap(func))(inputs),
        check_dtypes=False)
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      assertMultiLineStrippedEqual(
          self, """
          transforms: [('batch', {'batch_dims': (0,)})] where: before:x
          [0 1 2 3 4]
          transforms: [('batch', {'batch_dims': (0,)})] where: body:x+1
          [1 2 3 4 5]
          transforms: [('batch', {'batch_dims': (0,)})] where: body:x+1
          [2 3 3 4 5]
          transforms: [('batch', {'batch_dims': (0,)})] where: after:x
          [2 2 2 3 4]""", testing_stream.output)
    else:
      pass  # order of vmaps is not guaranteed

  def test_tap_vmap_while_tap_cond(self):
    """Vmap of while, with a tap in the conditional."""

    def func(x):
      # like max(x, 2)
      x1 = hcb_id_print(x, where="1", output_stream=testing_stream,
                        callback_flavor=hcb.CallbackFlavor.DEBUG)
      x2 = lax.while_loop(lambda x: hcb_id_print(x < 2, where="w_c",
                                                 output_stream=testing_stream,
                                                 callback_flavor=hcb.CallbackFlavor.DEBUG),
                          lambda x: hcb_id_print(x + 1, where="w_b",
                                                 output_stream=testing_stream,
                                                 callback_flavor=hcb.CallbackFlavor.DEBUG),
                          x1)
      res = hcb_id_print(x2, where="3", output_stream=testing_stream,
                         callback_flavor=hcb.CallbackFlavor.DEBUG)
      return res

    inputs = np.arange(5, dtype=np.int32)
    res = jax.jit(jax.vmap(func))(inputs)
    jax.effects_barrier()
    self.assertAllClose(np.array([2, 2, 2, 3, 4]), res, check_dtypes=False)
    if hcb._HOST_CALLBACK_LEGACY.value:
      assertMultiLineStrippedEqual(self, """
          transforms: [('batch', {'batch_dims': (0,)})] where: 1
          [0 1 2 3 4]
          transforms: [('batch', {'batch_dims': (0,)})] where: w_c
          [ True  True False False False]
          transforms: [('batch', {'batch_dims': (0,)})] where: w_b
          [1 2 3 4 5]
          transforms: [('batch', {'batch_dims': (0,)})] where: w_c
          [ True False False False False]
          transforms: [('batch', {'batch_dims': (0,)})] where: w_b
          [2 3 3 4 5]
          transforms: [('batch', {'batch_dims': (0,)})] where: w_c
          [False False False False False]
          transforms: [('batch', {'batch_dims': (0,)})] where: 3
          [2 2 2 3 4]""", testing_stream.output)
    else:
      pass  # order of vmap is not guaranteed

  def test_tap_transforms_doc(self):
    # Examples from the documentation
    def power3(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb_id_print((x, y), what="x,x^2", output_stream=testing_stream,
                   callback_flavor=hcb.CallbackFlavor.DEBUG)
      return y * x

    print(f"impl = {power3(3.)}")
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
            what: x,x^2
           ( 3. 9. )"""
    else:
      expected = """
            what: x,x^2
           ( 3.0 9.0 )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"jvp = {jax.jvp(power3, (3.,), (0.1,))}")
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
            what: x,x^2
           ( 3. 9. )"""
    else:
      expected = """
            what: x,x^2
           ( 3.0 9.0 )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    @jax.custom_jvp
    def print_tangents(arg):
      return None

    @print_tangents.defjvp
    def print_tangents_jvp(primals, tangents):
      arg_dot, = tangents
      hcb_id_print(arg_dot, what="tangents", output_stream=testing_stream,
                   callback_flavor=hcb.CallbackFlavor.DEBUG)
      return primals, tangents

    def power3_with_tangents(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb_id_print((x, y), what="x,x^2", output_stream=testing_stream,
                   callback_flavor=hcb.CallbackFlavor.DEBUG)
      print_tangents((x, y))
      return y * x

    print(f"jvp = {jax.jvp(power3_with_tangents, (3.,), (0.1,))}")
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
        what: x,x^2
        ( 3. 9. )
        what: tangents
        ( 0.1 0.6 )"""
      self.assertMultiLineStrippedEqual(expected, testing_stream.output)

    testing_stream.reset()

    print(f"grad = {jax.grad(power3)(3.)}")
    jax.effects_barrier()
    # Only the primals by default
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
         what: x,x^2
         ( 3. 9. )"""
    else:
      expected = """
         what: x,x^2
         ( 3.0 9.0 )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    @jax.custom_vjp
    def print_cotangents(arg):
      # Must return the argument for which we want the cotangent.
      return arg

    # f_fwd: a -> (b, residual)
    def print_cotangents_fwd(arg):
      return print_cotangents(arg), None
    # f_bwd: (residual, CT b) -> [CT a]
    def print_cotangents_bwd(residual, ct_b):
      hcb_id_print(ct_b, what="cotangents", output_stream=testing_stream,
                   callback_flavor=hcb.CallbackFlavor.DEBUG)
      return ct_b,

    print_cotangents.defvjp(print_cotangents_fwd, print_cotangents_bwd)

    def power3_with_cotangents(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb_id_print((x, y), what="x,x^2", output_stream=testing_stream,
                   callback_flavor=hcb.CallbackFlavor.DEBUG)
      # Must use the output of print_cotangents
      (x1, y1) = print_cotangents((x, y))
      return y1 * x1

    print(f"grad = {jax.grad(power3_with_cotangents)(3.)}")
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
        what: x,x^2
        ( 3. 9. )
        what: cotangents
        ( 9. 3. )"""
    else:
      expected = """
        what: x,x^2
        ( 3.0 9.0 )
        what: cotangents
        ( 9.0 3.0 )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    # TODO: grad of grad

    print(f"vmap = {jax.vmap(power3)(np.array([2., 3.]))}")
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
         transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
         ( [2. 3.] [4. 9.] )"""
    else:
      expected = """
        what: x,x^2
        ( 2.0 4.0 )
        what: x,x^2
        ( 3.0 9.0 )
        """
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"vmap o grad {jax.vmap(jax.grad(power3))(np.array([2., 3.]))}")
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
         transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
         ( [2. 3.] [4. 9.] )"""
    else:
      expected = """
        what: x,x^2
        ( 2.0 4.0 )
        what: x,x^2
        ( 3.0 9.0 )
        """
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"vmap o grad {jax.vmap(jax.grad(power3_with_cotangents))(np.array([2., 3.]))}")
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
        transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
        ( [2. 3.] [4. 9.] )
        transforms: [('batch', {'batch_dims': (0, 0)})] what: cotangents
        ( [4. 9.] [2. 3.] )"""
    else:
      expected = """
        what: x,x^2
        ( 2.0 4.0 )
        what: x,x^2
        ( 3.0 9.0 )
        what: cotangents
        ( 4.0 2.0 )
        what: cotangents
        ( 9.0 3.0 )
        """
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"grad o remat = {jax.grad(lambda x: power3(ad_checkpoint.checkpoint(power3)(x)))(3.)}")
    jax.effects_barrier()
    if hcb._HOST_CALLBACK_LEGACY.value:
      expected = """
        what: x,x^2
        ( 3. 9. )
        what: x,x^2
        ( 27. 729. )
        what: x,x^2
        ( 3. 9. )"""
    else:
      expected = """
        what: x,x^2
        ( 3.0 9.0 )
        what: x,x^2
        ( 27.0 729.0 )
        what: x,x^2
        ( 3.0 9.0 )
        """
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

  @unittest.skip("cond of pmap does not work in JAX. Issue #5178.")
  def test_tap_cond_pmap(self):
    # A matrix M[ij] = i * 10 + j
    nr_devices = len(local_devices())
    shape = (nr_devices, 3)
    matrix = np.fromfunction(lambda i, j: 10. * i + j, shape,
                             dtype=np.float32)

    def fun1(x, do_print=False):
      return maybe_print(do_print, x * 2., "x * 2")

    def fun2(cond, xv, do_print=False):
      return lax.cond(cond, jax.pmap(partial(fun1, do_print=do_print)),
                      lambda xv: xv, xv)

    res = fun2(True, matrix)
    self.assertAllClose(fun2(True, matrix, do_print=False), res, check_dtypes=False)
    jax.effects_barrier()
    assertMultiLineStrippedEqual(self, """
        TBD""", testing_stream.output)

  def test_tap_callback_delay(self):
    hcb.callback_extra = lambda dev: time.sleep(1)

    def func(x):
      for i in range(5):
        x = hcb_id_print(x * i, what="x times i")
      return x

    jax.jit(func)(np.arange(6, dtype=np.float32).reshape((2, 3)))

  def test_tap_callback_delay_barrier(self):
    hcb.callback_extra = lambda dev: time.sleep(2)

    def func(x):
      for i in range(1, 4):
        x = hcb_id_print(x * i, what=f"x times {i}", output_stream=testing_stream)
      return x

    jax.jit(func)(np.arange(6, dtype=np.float32).reshape((2, 3)))
    # Wait for the results
    jax.effects_barrier()
    expected = """
        what: x times 1
        [[0. 1. 2.]
        [3. 4. 5.]]
        what: x times 2
        [[ 0.  2.  4.]
        [ 6.  8. 10.]]
        what: x times 3
        [[ 0.  6. 12.]
        [18. 24. 30.]]"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()
    # Call again
    jax.jit(func)(np.arange(6, dtype=np.float32).reshape((2, 3)))
    jax.effects_barrier()
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_tap_error_bad_consumer_id(self):
    """Try to use reserved consumer ID 0.

    Check that we get the proper error from the runtime."""
    if not hcb._use_outfeed(jtu.device_under_test()):
      raise SkipTest("test works only for outfeed")
    comp = xla_client.XlaBuilder(self._testMethodName)
    token = hcb.xops.CreateToken(comp)
    hcb._initialize_outfeed_receiver()  # Needed if this is the sole test
    with self.assertRaisesRegex(RuntimeError,
                                "Consumer ID cannot be a reserved value: 0"):
      hcb._callback_handler_data.receiver.add_outfeed(
          comp, token, 0,
          [xops.Constant(comp, np.zeros((2, 3), dtype=np.float32))], 0)

  def test_tap_error_different_shapes(self):
    """Try to register different shapes for the same consumer ID."""
    if not hcb._use_outfeed(jtu.device_under_test()):
      raise SkipTest("test works only for outfeed")
    comp = xla_client.XlaBuilder(self._testMethodName)
    token = hcb.xops.CreateToken(comp)
    hcb._initialize_outfeed_receiver()  # Needed if this is the sole test
    hcb._callback_handler_data.receiver.add_outfeed(
        comp, token, 123,
        [xops.Constant(comp, np.zeros((2, 3), dtype=np.float32))], 0)
    with self.assertRaisesRegex(
        RuntimeError, ".*does not match previous shape .*\n?element_type.*"):
      hcb._callback_handler_data.receiver.add_outfeed(
          comp, token, 123,
          [xops.Constant(comp, np.zeros((2, 3), dtype=np.int32))], 0)
    with self.assertRaisesRegex(
        RuntimeError, ".*does not match previous shape .*\n?element_type.*"):
      hcb._callback_handler_data.receiver.add_outfeed(
          comp, token, 123,
          [xops.Constant(comp, np.zeros((2,), dtype=np.float32))], 0)

  def test_tap_odeint(self):
    # TODO: find a smaller repro for bug #4015
    # Seems to be xla_call(scan(xla_call)), all under grad.
    from jax.experimental.ode import odeint

    def f(x, t, k):
      x = hcb_id_print(x, callback_flavor=hcb.CallbackFlavor.DEBUG)
      return -k * x

    def loss(k=1.0):
      t = jnp.linspace(0, 0.001, num=2)
      xs = odeint(f, 1.0, t, k)
      return xs[-1]

    jax.grad(loss)(1.0)  # should not fail

  def test_tap_remat_0(self):
    def f(i, k):
      x = hcb_id_print(k + i, output_stream=testing_stream,
                       callback_flavor=hcb.CallbackFlavor.DEBUG)
      return k * x

    def loss(k):
      return lax.fori_loop(0, 2, jax.remat(f), k)

    print(loss(3))
    jax.effects_barrier()
    expected = """
      3
      10"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_tap_named_call(self):
    def tap_scalar(init, do_print=False):
      @partial(jax.named_call, name="step")
      def step(acc, step_nr):
        acc = acc + step_nr
        maybe_print(do_print, step_nr, what="step_nr")
        return acc, None

      return lax.scan(step, init, np.arange(2))

    self.assertAllClose(tap_scalar(3, do_print=False), tap_scalar(3, do_print=True))
    jax.effects_barrier()
    expected = """
      what: step_nr
      0
      what: step_nr
      1"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)


class HostCallbackCallTest(jtu.JaxTestCase):
  """Tests for hcb.call"""

  def setUp(self):
    # skipping here skips teardown, so do this before super().setUp().
    if jtu.test_device_matches(["gpu"]) and jax.device_count() > 1:
      raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")
    if xla_bridge.using_pjrt_c_api():
      raise SkipTest("host_callback not implemented in PJRT C API")
    super().setUp()
    self.enter_context(jtu.ignore_warning(
      category=DeprecationWarning, message="The host_callback APIs are deprecated"))
    self.enter_context(jtu.ignore_warning(
      category=DeprecationWarning, message="backend and device argument"))

    testing_stream.reset()
    testing_stream._test_method_name = self._testMethodName

  def tearDown(self) -> None:
    jax.effects_barrier()
    super().tearDown()

  def call_log_testing_stream(self, func, arg, *, result_shape, name=""):
    """Call `func` and log inputs and outputs to the testing stream"""

    def call_log(arg):
      def val2str(v):
        return np.array2string(np.array(arg))
      testing_stream.write(f"Call {name}({val2str(arg)})\n")
      res = func(arg)
      testing_stream.write(f"  = {val2str(res)}\n")
      return res
    return hcb.call(call_log, arg, result_shape=result_shape)

  def test_call_simple(self):

    def f_outside(x):
      return 2 * x

    def fun(x):
      y = hcb.call(f_outside, x + 1, result_shape=x)
      return 3 * (1 + y)

    arg = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    self.assertAllClose(3 * (1 + 2 * (arg + 1)), fun(arg))

  def test_primitive_compilation(self):

    def f_outside(x):
      return 2 * x

    def fun(x):
      return hcb.call(f_outside, x, result_shape=x)

    arg = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    with jtu.count_primitive_compiles() as count:
      for _ in range(3):
        self.assertAllClose(2 * arg, fun(arg))
    r = jax.make_jaxpr(fun)(arg)
    self.assertEqual(count[0], 1)

  @jtu.sample_product(
    dtype=[dtype for dtype in jtu.dtypes.all if dtype != np.bool_],
  )
  def test_call_types(self, dtype=np.float64):

    def f_outside(x):
      # Use x + x to ensure that the result type is the same
      return x + x

    def fun(x):
      return hcb.call(f_outside, x + x, result_shape=x)

    arg = np.arange(24, dtype=dtype).reshape((2, 3, 4))
    self.assertAllClose(arg + arg + arg + arg, fun(arg), check_dtypes=True)

  def test_call_types_bool(self, dtype=np.float64):

    def f_outside(x):
      return np.invert(x)

    def fun(x):
      return hcb.call(f_outside, x, result_shape=x)

    arg = self.rng().choice(a=[True, False], size=(2, 3, 4))
    self.assertAllClose(np.invert(arg), fun(arg))

  def test_call_tuples(self):

    def f_outside(args):
      x, y = args
      return y, x  # Swap the tuple

    def fun(x):
      xy = hcb.call(f_outside, (x, x + 1), result_shape=(x, x))
      return 2 * xy[0] + 3 * xy[1]

    arg = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    self.assertAllClose(2 * (arg + 1) + 3 * arg, fun(arg))

  def test_call_no_arg(self):
    """Call with no arguments."""
    result = np.ones((2,), dtype=np.float32)
    def f_outside(in_tuple):
      assert len(in_tuple) == 0
      return result
    def fun(x):
      return x + hcb.call(f_outside, (),
                          result_shape=jax.ShapeDtypeStruct(result.shape, result.dtype))
    self.assertAllClose(2. + result, fun(2.))

  def test_call_empty_arg(self):
    """Call with empty array."""
    result = np.full((2,), 3., dtype=np.float32)
    def f_outside(x0):  # x0: f32[2, 0]
      return result
    x0 = np.ones((2, 0), dtype=np.float32)
    def fun(x):
      return x + hcb.call(f_outside, x0,
                          result_shape=jax.ShapeDtypeStruct(result.shape, result.dtype))
    self.assertAllClose(2. + result, fun(2.))

  def test_call_empty_arg_inside_pytree(self):
    """Call taking tuple with an empty array and a non-empty one."""
    x0 = np.ones((2, 0), dtype=np.float32)
    x1 = np.full((2,), 3., dtype=np.float32)
    result = x1
    def f_outside(in_tuple):  # x0: f32[2, 0]  x1: f32[2]
      return in_tuple[1]

    def fun(x):
      res = hcb.call(f_outside, (x0, x1),
                     result_shape=jax.ShapeDtypeStruct(result.shape, result.dtype))
      return x + res
    self.assertAllClose(2. + result, fun(2.))

  def test_call_empty_result(self):
    """Call returning empty array."""
    result_shape = (2, 0)
    def f_outside(_):
      return np.ones(result_shape, dtype=np.float32)
    def fun(x):
      return x + hcb.call(f_outside, 1.,
                          result_shape=jax.ShapeDtypeStruct(result_shape, np.float32))
    self.assertAllClose(f_outside(0.), fun(2.))

  def test_call_empty_result_inside_pytree(self):
    """Call returning a tuple with an empty array and a non-empty one."""
    result_shape_0 = (2, 0)
    result_shape_2 = (0,)
    def f_outside(_):
      return (np.ones(result_shape_0, dtype=np.float32),
              np.ones((1,), dtype=np.float32),
              np.ones(result_shape_2, dtype=np.float32))
    def fun(x):
      res = hcb.call(f_outside, 1.,
                     result_shape=(jax.ShapeDtypeStruct(result_shape_0, np.float32),
                                   jax.ShapeDtypeStruct((1,), np.float32),
                                   jax.ShapeDtypeStruct(result_shape_2, np.float32)))
      self.assertEqual(result_shape_0, res[0].shape)
      self.assertEqual(result_shape_2, res[2].shape)
      return x + res[1]
    self.assertAllClose(2 + np.ones((1,), dtype=np.float32), fun(2.))

  def test_call_empty_result_all_pytree(self):
    """Call returning a tuple of empty arrays."""
    result_shape = (2, 0)
    def f_outside(_):
      return (np.ones(result_shape, dtype=np.float32),
              np.ones(result_shape, dtype=np.float32))
    def fun(x):
      res = hcb.call(f_outside, 1.,
                     result_shape=(jax.ShapeDtypeStruct(result_shape, np.float32),
                                   jax.ShapeDtypeStruct(result_shape, np.float32)))
      return x + res[0] + res[1]
    self.assertAllClose(np.ones(result_shape, dtype=np.float32),
                        fun(2.))

  def test_call_no_result(self):
    def f_outside(arg):
      self.call_log_testing_stream(lambda x: None, arg,
                                   result_shape=None,
                                   name="outside")
      return arg

    self.assertAllClose((3., 4.), f_outside((3., 4.)))
    jax.effects_barrier()
    expected = """
        Call outside([3. 4.])
          = [3. 4.]"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_call_cond(self):
    def f_outside(args):
      x, y = args
      return x * y.astype(np.float32)

    def loop(x, use_outside=True):
      def body(i, acc):
        return lax.cond(i % 2 == 1,
                        lambda _: (hcb.call(f_outside, (acc, i),
                                            result_shape=acc)
                                   if use_outside else f_outside((acc, i))),
                        lambda _: acc,
                        None)

      return lax.fori_loop(0, 18, body, x)

    res_inside = loop(np.float32(1.2), use_outside=False)
    self.assertAllClose(res_inside, jax.jit(loop)(np.float32(1.2)))

  def test_call_jit_scan_call(self):
    def f_outside(x):
      return x

    def loop(x, use_outside=True):
      def body(carry, i):
        if use_outside:
          return carry + hcb.call(f_outside, i,
                                  result_shape=i), None
        else:
          return carry + i, None

      return lax.scan(body, 0, x)

    x = np.arange(5, dtype=np.int32)

    res_outside = jax.jit(partial(loop, use_outside=True))(x)
    self.assertAllClose(res_outside, loop(x, use_outside=False))

  def test_call_doc_example1(self):
    """Examples from the documentation: simplest, call a function"""

    def host_eig(x):
      return np.linalg.eigvals(x)

    shape = (2, 5, 4, 4)

    m = np.ones(shape, dtype=np.float32)

    def fun(m):
      eig_m = hcb.call(host_eig, m,
                       result_shape=jax.ShapeDtypeStruct(m.shape[:-1], m.dtype))
      return eig_m

    expected_res = np.linalg.eigvals(m)
    self.assertAllClose(expected_res, fun(m))
  @jtu.skip_on_devices("gpu")
  def test_call_doc_example_hlo(self):
    """Examples from the documentation: simplest, call a function."""

    def fun1(m):
      return jnp.sin(hcb.call(lambda x: np.cos,
                              jnp.cos(m),
                              result_shape=m))

    m = np.ones((2,), np.float32)
    helper_print_optimized_hlo(fun1, m)

    def fun2(m):
      x = hcb.call(lambda x: None, 2, result_shape=())
      return x

    m = np.ones((2,), np.float32)
    helper_print_optimized_hlo(fun2, m)

  def test_call_vmap(self):
    def f_outside(x): return x

    def fun(x):
      return hcb.call(f_outside, x, result_shape=x,
                      callback_flavor=hcb.CallbackFlavor.PURE)

    if hcb._HOST_CALLBACK_LEGACY.value:
      with self.assertRaisesRegex(NotImplementedError,
                                  "batching rules are implemented only for id_tap, not for call"):
        jax.vmap(fun)(np.ones((2, 3)))
    else:
      with jtu.ignore_warning(category=DeprecationWarning):
        jax.vmap(fun)(np.ones((2, 3)))

  def test_call_error_bad_result_shape(self):
    with self.assertRaisesRegex(
        ValueError,
        "The values must be either numeric scalars, or must have 'shape' and 'dtype' attributes"):
      hcb.call(lambda x: x, 3., result_shape="string")

    with self.assertRaisesRegex(
        ValueError,
        "The values must be either numeric scalars, or must have 'shape' and 'dtype' attributes"):
      hcb.call(lambda x: x, 3., result_shape=lambda x: x)
      jax.effects_barrier()

  def helper_check_callback_errors(self, thunk: Callable,
                                   expected_exc_txt: str):
    """Calls thunk() and checks for expected exceptions.
    """
    if jtu.test_device_matches(["cpu"]):
      # On CPU the runtime crashes, and the tests are all aborted
      raise SkipTest("TODO: CPU runtime crashes on unexpected infeed")
    elif jtu.test_device_matches(["gpu"]):
      # On GPU we get a nice error back to Python
      with self.assertRaisesRegex(
          RuntimeError,
          "(.* Mismatch between infeed source buffer shape s8.12345."
          "|.*The destination shape does not match the source shape.)"):
        thunk()
    elif jtu.test_device_matches(["tpu"]):
      # On TPU we get no error!!!
      raise SkipTest("TODO: TPU runtime does not check infeed, and just computes with garbage")

    # Both on GPU and TPU we also get an error during the barrier_wait at the
    # end of the test. Run a barrier_wait now, to consume that error.
    with self.assertRaisesRegex(
        hcb.CallbackException,
        re.compile(
            "There were exceptions during callback processing.*Last one was:.*" +
            expected_exc_txt,
            re.DOTALL)):
      jax.effects_barrier()


def call_jax_other_device(
    jax_outside_fun, arg, *, device,
    callback_flavor: hcb.CallbackFlavor = hcb.CallbackFlavor.IO_CALLBACK):
  """Calls a JAX function on a specific device with simple support for reverse AD.

  Functions whose name starts with "jax_outside" are called on another device,
  by way of hcb.call.
  """

  def run_jax_outside_fun(arg):
    return jax.jit(jax_outside_fun)(jax.device_put(arg, device))

  @jax.custom_vjp
  def make_call(arg):
    return hcb.call(run_jax_outside_fun, arg,
                    result_shape=jax.eval_shape(jax_outside_fun, arg),
                    callback_flavor=callback_flavor)

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(arg):
    # Return the primal argument as the residual. Use `make_call` for the
    # primal computation to enable higher-order AD.
    return make_call(arg), arg  # Return the primal argument as the residual

  def make_call_vjp_bwd(res, ct_res):
    arg = res  # residual is the primal argument

    def jax_outside_vjp_fun(arg_and_ct):
      arg, ct = arg_and_ct
      _, f_vjp = jax.vjp(jax_outside_fun, arg)
      ct_in, = f_vjp(ct)
      return ct_in

    return (call_jax_other_device(jax_outside_vjp_fun, (arg, ct_res), device=device),)

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return make_call(arg)


class CallJaxTest(jtu.JaxTestCase):
  """Tests using `call_jax_other_device`."""

  def setUp(self):
    if not hcb._HOST_CALLBACK_LEGACY.value:
      self.skipTest("Not supported when JAX_HOST_CALLBACK_LEGACY=False")
    if jtu.test_device_matches(["gpu"]) and jax.device_count() > 1:
      raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")
    if xla_bridge.using_pjrt_c_api():
      raise SkipTest("host_callback not implemented in PJRT C API")

    if not jtu.test_device_matches(["cpu"]):
      assert jax.devices("cpu")
      self.outside_device = jax.devices("cpu")[0]
    else:
      if len(jax.devices("cpu")) == 1:
        raise SkipTest("Test needs at least two devices. On CPU use XLA_FLAGS=--xla_force_host_platform_device_count=2")
      self.outside_device = jax.devices("cpu")[1]
    super().setUp()
    self.enter_context(jtu.ignore_warning(
      category=DeprecationWarning, message="The host_callback APIs are deprecated"))


  def test_jax_impl(self):
    def f_jax(x):
      return jnp.sin(x)

    def f_outside(x):
      return call_jax_other_device(f_jax, x, device=self.outside_device)

    self.assertAllClose(f_jax(3.), f_outside(3.))
    self.assertAllClose(f_jax(3.), jax.jit(f_outside)(3.))

  def test_jax_impl_pytree(self):
    def f_jax(x):
      # x : dict(a=..., b=...) and output is a list of two elements
      return [jnp.sin(x["a"]), jnp.sin(x["b"])]

    def f_outside(x):
      return call_jax_other_device(f_jax, x, device=self.outside_device)

    x = dict(a=3., b=4.)
    res_jax = f_jax(x)
    # print(f"outside_jaxpr = {jax.make_jaxpr(f_outside)(x)}")
    res_outside = f_outside(x)
    self.assertAllClose(res_jax, res_outside)

  def test_jax_grad(self):
    def f_jax(x):
      return 2. * jnp.sin(x)

    def f_outside(x):
      return 2. * call_jax_other_device(jnp.sin, x, device=self.outside_device)

    res_jax = jax.grad(f_jax)(3.)
    self.assertAllClose(res_jax, jax.grad(f_outside)(3.))

  def test_jax_grad_pytree(self):
    def f_jax(x):
      # x : dict(a=..., b=...) and output is a float
      return 3. * jnp.sin(x["a"]) + jnp.sin(x["b"])

    def f_outside(x):
      return call_jax_other_device(f_jax, x, device=self.outside_device)

    x = dict(a=3., b=4.)
    res_jax = jax.grad(f_jax)(x)
    self.assertAllClose(res_jax, jax.grad(f_outside)(x))

  def test_jax_grad_of_grad(self):
    def f_jax(x):
      return 2. * x * x * x

    def f_outside(x):
      return 2. * call_jax_other_device(lambda x: x * x * x, x, device=self.outside_device)

    res_jax = jax.grad(jax.grad(f_jax))(5.)
    res_outside = jax.grad(jax.grad(f_outside))(5.)
    self.assertAllClose(res_jax, res_outside)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
