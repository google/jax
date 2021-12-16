# Copyright 2020 Google LLC
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

from functools import partial
import itertools
import logging
import os
import re
import threading
import time
from typing import Callable, Optional, Sequence
import unittest
from unittest import skip, SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import ad_checkpoint
from jax import core
from jax.config import config
from jax import dtypes
from jax.experimental import host_callback as hcb
from jax.experimental import PartitionSpec as P
from jax.experimental import maps
from jax.interpreters import partial_eval as pe
from jax.experimental import pjit
from jax.interpreters import xla
from jax import lax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax import tree_util
from jax._src.lib import xla_client
from jax._src.lib import xla_bridge

xops = xla_client.ops

import numpy as np

config.parse_flags_with_absl()
FLAGS = config.FLAGS


class _TestingOutputStream(object):
  """Use as `output_stream` for tests."""

  def __init__(self):
    self._output = []
    self._test_method_name = None

  def write(self, what: str) -> None:
    print(f"output_stream[{self._test_method_name}]: {what}", end="")
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
  y = hcb.id_print(a * 2., what="a * 2", output_stream=testing_stream)
  y = hcb.id_print(y * 3., what="y * 3", output_stream=testing_stream, result=y)
  return y ** 2  # Some computation to make the gradient interesting


def fun1_equiv(a):  # Numerical equivalent of fun1
  return (a * 2.) ** 2


def maybe_print(do_print: bool, arg, what: str, tap_with_device: Optional[bool] = False):
  """Conditionally print on testing_string"""
  if do_print:
    return hcb.id_print(arg, what=what,
                        output_stream=testing_stream, tap_with_device=tap_with_device)
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

  what = re.sub(r"\-?\d*\.[\-\def]*", repl_floats, what)
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
  backend = xla_bridge.get_backend()
  c = jax.xla_computation(fun, backend='cpu')(*args)
  print(re.sub(r", metadata.*", "",
               backend.compile(c).hlo_modules()[0].to_string()))


def helper_log_ir(name,
                  f_jax,
                  *args,
                  num_partitions=None,
                  strip_metadata=False):
  print(f"Jaxpr[{name}]: {jax.make_jaxpr(f_jax)(*args)}")
  jax_comp = jax.xla_computation(f_jax, backend='cpu')(*args)
  print(f"HLO[{name}]: {jax_comp.as_hlo_text()}")

  backend = xla_bridge.get_backend()
  if num_partitions is not None:
    num_replicas = 1
    device_assignment = np.arange(num_partitions * num_replicas)
    device_assignment = np.reshape(device_assignment, (-1, num_partitions))
    use_spmd_partitioning = num_partitions > 1
    compile_options = xla_bridge.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=use_spmd_partitioning,
    )
  else:
    compile_options = None
  jax_optimized_hlo = backend.compile(
      jax_comp, compile_options).hlo_modules()[0].to_string()
  if strip_metadata:
    jax_optimized_hlo = re.sub(r", metadata.*", "", jax_optimized_hlo)
  print(f"Optimized HLO[{name}] for "
               f"platform {backend.platform}: {jax_optimized_hlo}")


prev_xla_flags = None


def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)


# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()


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


class HostCallbackTapTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() == "gpu" and jax.device_count() > 1:
      raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")

    testing_stream.reset()
    testing_stream._test_method_name = self._testMethodName
    self.old_flags = os.getenv("XLA_FLAGS", "")

  def tearDown(self) -> None:
    if os.getenv("XLA_FLAGS") != self.old_flags:
      os.environ["XLA_FLAGS"] = self.old_flags
      xla_bridge.get_backend.cache_clear()
    hcb.barrier_wait("HostCallbackTapTest.tearDown")
    super().tearDown()

  def test_tap_eval(self):
    self.assertAllClose((5. * 2.) ** 2, fun1(5.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        what: a * 2
        10.00
        what: y * 3
        30.00""", testing_stream.output)

  def test_tap_with_tuple_results(self):
    def func2(x):
      x1, y1 = hcb.id_print((x * 2., x * 3.), output_stream=testing_stream)
      return x1 + y1

    self.assertEqual(3. * (2. + 3.), func2(3.))
    hcb.barrier_wait()

    assertMultiLineStrippedEqual(self, """
        ( 6.00 9.00 )""", testing_stream.output)

  def test_tap_with_dict_results(self):
    def func2(x):
      res = hcb.id_print(dict(a=x * 2., b=x * 3.), output_stream=testing_stream)
      return res["a"] + res["b"]

    self.assertEqual(3. * (2. + 3.), func2(3.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        { a=6.00 b=9.00 }""", testing_stream.output)

  def test_tap_with_result(self):
    def func2(x):
      x1 = hcb.id_print((x * 2., x * 3.), result=x * 4.,
                        output_stream=testing_stream)
      return x1

    self.assertEqual(3. * 4., func2(3.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        ( 6.00 9.00 )""", testing_stream.output)

  def test_tap_with_result_no_arg(self):
    def tap_func(arg, transforms):
      testing_stream.write(f"called tap_func with {arg}")

    def func2(x):
      x1 = hcb.id_tap(tap_func, None, result=x)
      return x1

    self.assertEqual(3., func2(3.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, "called tap_func with None",
                                 testing_stream.output)

  def test_tap_result_unused(self):
    def tap_func(arg, transforms):
      testing_stream.write(f"called tap_func with {arg}")
    def func2(x):
      hcb.id_tap(tap_func, None)
      return x

    self.assertEqual(3., func2(3.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, "called tap_func with None",
                                 testing_stream.output)

  def test_tap_with_device(self):
    def func2(x):
      x1 = hcb.id_print((x * 2., x * 3.), result=x * 4.,
                        output_stream=testing_stream,
                        tap_with_device=True)
      return x1

    self.assertEqual(3. * 4., func2(3.))
    hcb.barrier_wait()
    assertMultiDeviceOutputEqual(self, """
      device: cpu:0
      ( 6.00 9.00 )""")

  def test_tap_eval_exception(self):
    if not FLAGS.jax_host_callback_outfeed:
      raise SkipTest("TODO: implement error handling for customcall")
    # Simulate a tap error
    def tap_err(*args, **kwargs):
      raise ValueError("Some user message")

    def func(x):
      x1 = hcb.id_print(x + 1, what="x1", output_stream=testing_stream)
      x2 = hcb.id_tap(tap_err, x1 + 1)
      x3 = hcb.id_print(x2 + 1, what="x3", output_stream=testing_stream)
      return x3

    with self.assertRaisesRegex(
       hcb.CallbackException,
       re.compile("There were exceptions during callback processing. Last one was:.*"
                  "ValueError: Some user message", re.DOTALL)):
      func(0)
      hcb.barrier_wait()

    # We should have received everything before the error
    assertMultiLineStrippedEqual(self, """
        what: x1
        1
        what: x3
        3""", testing_stream.output)

  def test_tap_empty(self):
    """Tap empty arrays."""
    hcb.id_print((), output_stream=testing_stream)
    hcb.id_print((1., np.ones((2, 0))), what="second", output_stream=testing_stream)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        (  )
        what: second
        ( 1.00 [] )""", testing_stream.output)

  def test_tap_jit_simple(self):
    jit_fun1 = jax.jit(lambda x: 3. * hcb.id_print(
        2. * x, what="here", output_stream=testing_stream))
    self.assertAllClose(6. * 5., jit_fun1(5.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        what: here
        10.00""", testing_stream.output)

  def test_tap_jit_no_invars(self):
    def func():  # jitted function does not take arguments
      return hcb.id_print(42, output_stream=testing_stream)

    self.assertAllClose(42, jax.jit(func)())
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
    42""", testing_stream.output)

  def test_tap_jit_multiple_invars(self):
    def func(x1, x2):
      return hcb.id_print(x1 + x2, output_stream=testing_stream)

    self.assertAllClose(42, jax.jit(func)(40, 2))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
    42""", testing_stream.output)

  def test_tap_jit_constant(self):
    def func(x):
      return hcb.id_print(42, result=x, output_stream=testing_stream)

    self.assertAllClose(5, jax.jit(func)(5))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
    42""", testing_stream.output)

  def test_tap_jit_sequence1(self):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      return hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

    logging.info("%s: %s", self._testMethodName,
                 jax.make_jaxpr(func)(1))
    logging.info("%s: %s", self._testMethodName,
                 jax.xla_computation(func, backend='cpu')(1).as_hlo_text())
    self.assertEqual(2, jax.jit(func)(1))
    hcb.barrier_wait()

    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: 2
        2""", testing_stream.output)

  def test_tap_jit2(self):
    """A sequence of JIT."""

    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)
      return x2

    self.assertEqual(2, jax.jit(func)(1))
    self.assertEqual(11, jax.jit(func)(10))
    hcb.barrier_wait()
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
      hcb.id_print(x, where="1", output_stream=testing_stream)
      hcb.id_print(x + 1, where="2", output_stream=testing_stream)
      return x + 1

    self.assertEqual(2, jax.jit(func)(1))
    self.assertEqual(11, jax.jit(func)(10))
    hcb.barrier_wait()
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
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)

      def func_nested(x):
        x2 = hcb.id_print(x + 1, where="nested", output_stream=testing_stream)
        return x2

      x3 = jax.jit(func_nested)(x1)
      return hcb.id_print(x3 + 1, where="3", output_stream=testing_stream)

    self.assertEqual(3, jax.jit(func)(1))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: nested
        2
        where: 3
        3""", testing_stream.output)

  def test_tap_jit_devices(self):
    """Running on multiple devices."""
    logging.info("%s: has devices %s", self._testMethodName, local_devices())

    def func(x, device_id):
      x1 = hcb.id_print(x, dev=str(device_id), output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, dev=str(device_id), output_stream=testing_stream)
      return x2

    for d in local_devices():
      self.assertEqual(112, jax.jit(func, device=d, static_argnums=1)(111, d.id))
    hcb.barrier_wait()
    logging.info("%s: found output %s", self._testMethodName,
                 testing_stream.output)
    self.assertEqual(
        len(local_devices()), len(re.findall(r"111", testing_stream.output)))
    self.assertEqual(
        len(local_devices()), len(re.findall(r"112", testing_stream.output)))

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_with_jit_{with_jit}",
              with_jit=with_jit)
          for with_jit in [True, False]))
  def test_tap_pytree(self, with_jit=False):
    def func(x, what=""):
      """Returns some pytrees depending on x"""
      if what == "pair_1_x":
        return (1, x)
      elif what == "pair_x_2x":
        return (x, 2 * x)
      elif what == "dict":
        return dict(a=2 * x, b=3 * x)
      else:
        assert False

    tap_count = 0

    def tap_func(a, _, *, what=""):
      nonlocal tap_count
      tap_count += 1
      self.assertEqual(func(5, what), a)

    transform = jax.jit if with_jit else lambda f: f
    for what in ("pair_1_x", "pair_x_2x", "dict"):
      transformed = transform(
          lambda x: hcb.id_tap(
              partial(tap_func, what=what),
              func(x, what),
              result=func(x * 2, what))
      )(5)
      self.assertEqual(func(10, what), transformed)
    hcb.barrier_wait()  # Wait for receivers to be done
    self.assertEqual(3, tap_count)

  def test_tap_xla_computation(self):
    def fun(x):
      return hcb.id_print(x * 2, output_stream=testing_stream)

    # Ensure that xla_computation has the tokens
    res = jax.xla_computation(fun)(1.).as_hlo_text()
    self.assertIn("token", res)

  def test_tap_lower_fun(self):
    class Prim(jax.core.Primitive):
      def __init__(self):
        super(Prim, self).__init__("prim")
        self.multiple_results = True

        def _abstract(*flat_avals, **params):
          return pe.abstract_eval_fun(self.impl, *flat_avals, **params)

        self.def_abstract_eval(_abstract)

        def _xla(c, *xla_args, **params):
          translation = xla.lower_fun(self.impl, multiple_results=True,
                                      backend="cpu")
          return translation(c, *xla_args, **params)

        xla.translations[self] = _xla

      def impl(self, *args):
        x, = args
        ret = hcb.id_print(x * 3., output_stream=testing_stream)
        return ret,

    @jax.jit
    def fn(x):
      return Prim().bind(x)[0]

    res = fn(jnp.asarray(1.))
    self.assertEqual(res, 3.)
    hcb.barrier_wait()  # Wait for receivers to be done
    assertMultiLineStrippedEqual(self, """
        3.00""", testing_stream.output)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_concurrent_{concurrent}",
              concurrent=concurrent)
          for concurrent in [True, False]))
  def test_tap_multiple(self, concurrent=False):
    """Call id_tap multiple times, concurrently or in sequence. """
    if concurrent and jtu.device_under_test() in ["cpu", "gpu"]:
      # TODO(necula): if there is device side concurrency, outfeeds from
      # different computations can be interleaved. For example, it seems that
      # on GPU if multiple host threads run a jit computation, the multiple
      # computations are interleaved on the GPU. This can result in the outfeed
      # trains being interleaved, which will trigger an error.
      # The solution is to fix on GPU the receiving logic so that we can outfeed
      # the train as one tuple, and receive it one piece as a time. Then the
      # trains should be atomic.
      # See also b/160692602.
      raise SkipTest("concurrent id_tap not supported on CPU, GPU")

    received = set()
    count = 5

    def pause_tap(idx, _):
      received.add(int(idx))
      logging.info("Starting do_tap %s. Sleeping 1sec ...", idx)
      time.sleep(0.3)
      logging.info("Finish do_tap %s", idx)

    def do_tap(idx):
      jax.jit(lambda idx: hcb.id_tap(pause_tap, idx))(idx)

    if concurrent:
      threads = [
          threading.Thread(
              name=f"enqueue_tap_{idx}", target=do_tap, args=(idx,))
          for idx in range(count)
      ]
      [t.start() for t in threads]
      [t.join() for t in threads]
    else:
      for idx in range(count):
        do_tap(idx)

    hcb.barrier_wait()
    self.assertEqual(received, set(range(count)))

  # TODO(necula): see comment for test_multiple_tap. Here we disable also
  # on TPU, because the barrier_wait runs on all devices, including on the CPU
  # where it would run into concurrency problems.
  @skip("Concurrency not supported")
  def test_tap_multiple_barriers(self):
    """Call barrier_wait concurrently."""

    def pause_tap(*args, **kwargs):
      logging.info("pause_tap waiting")
      time.sleep(0.3)
      logging.info("pause_tap done")

    def long_run(x):
      return hcb.id_tap(pause_tap, x)

    jax.jit(long_run)(5.)

    def try_barrier(idx):
      logging.info("Starting test barrier %s", idx)
      hcb.barrier_wait()
      logging.info("Finished test barrier %s", idx)

    threads = [
        threading.Thread(
            name=f"barrier_{idx}", target=try_barrier, args=(idx,))
        for idx in range(3)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_with_jit_{with_jit}",
              with_jit=with_jit)
          for with_jit in [True, False]))
  def test_tap_cond(self, with_jit=False):
    """A conditional"""

    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

      x4 = lax.cond(x % 2 == 0,
                    lambda x: hcb.id_print(x, where="cond_t",
                                           output_stream=testing_stream),
                    lambda x: hcb.id_print(-1, where="cond_f", result=x,
                                           output_stream=testing_stream),
                    x2 + 1)
      x5 = hcb.id_print(x4 + 1, where="end", output_stream=testing_stream)
      return x5

    transform = jax.jit if with_jit else lambda f: f
    self.assertEqual(4, transform(func)(1))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        where: 1
        1
        where: 2
        2
        where: cond_f
        -1
        where: end
        4""", testing_stream.output)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(testcase_name=f"_with_jit_{with_jit}",
               with_jit=with_jit)
          for with_jit in [True, False]))
  def test_tap_while_cond(self, with_jit=False):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

      def body(x):
        x3 = hcb.id_print(x, where="w_b_1", output_stream=testing_stream)
        x4 = lax.cond(x % 2 == 0,
                      lambda x: hcb.id_print(x, where="w_b_t",
                                             output_stream=testing_stream),
                      lambda x: hcb.id_print(-1, where="w_b_f",
                                             result=x, output_stream=testing_stream),
                      x3 + 1)
        return hcb.id_print(x4, where="w_b_2", output_stream=testing_stream)

      x10 = lax.while_loop(lambda x: x <= 3, body, x2)
      res = hcb.id_print(x10, where="end", output_stream=testing_stream)
      return res

    transform = jax.jit if with_jit else lambda f: f
    self.assertEqual(4, transform(func)(1))
    hcb.barrier_wait()
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
      x1 = hcb.id_print(x, where="1")
      x10 = lax.while_loop(lambda x: hcb.id_print(x < 3,
                                                  where="w_p",
                                                  output_stream=testing_stream),
                           lambda x: hcb.id_print(x + 1, where="w_b",
                                                  output_stream=testing_stream),
                           x1)
      res = hcb.id_print(x10, where="3", output_stream=testing_stream)
      return res

    self.assertEqual(3, jax.jit(func)(1))
    hcb.barrier_wait()
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_with_jit_{with_jit}",
              with_jit=with_jit)
          for with_jit in [True, False]))
  def test_tap_scan_cond(self, with_jit=True):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

      def body(c, x):
        x3 = hcb.id_print(x, where="s_1", output_stream=testing_stream)
        x4 = lax.cond(x % 2 == 0,
                      lambda x: hcb.id_print(x, where="s_t", output_stream=testing_stream),
                      lambda x: hcb.id_print(-1, where="s_f", result=x, output_stream=testing_stream),
                      x3 + 1)
        return (c, hcb.id_print(x4, where="s_2", output_stream=testing_stream))

      _, x10 = lax.scan(body, x2, jnp.arange(3))
      res = hcb.id_print(x10, where="10", output_stream=testing_stream)
      return res

    if with_jit:
      func = jax.jit(func)
    res = func(1)
    self.assertAllClose(jnp.array([1, 2, 3]), res)
    hcb.barrier_wait()
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_shape_{shape}_dtype_{np.dtype(dtype).name}_nr_args={nr_args}",
              shape=shape,
              dtype=dtype,
              nr_args=nr_args) for nr_args in [1, 2]
          for shape in [(), (2,), (2, 3), (2, 3, 4)]
          for dtype in jtu.dtypes.all))
  def test_tap_jit_dtypes(self, nr_args=2, dtype=jnp.int16, shape=(2,)):
    if dtype in (jnp.complex64, jnp.complex128, jnp.bool_):
      raise SkipTest(f"host_callback not implemented for {dtype}.")
    if dtype == np.bool_:
      args = [self.rng().choice(a=[True, False], size=shape)]
    else:
      args = [jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)]
    if nr_args > 1:
      args = args * nr_args
    jit_fun1 = jax.jit(lambda xs: hcb.id_print(
        xs,
        a_new_test="************",
        testcase_name=f"shape_{shape}_dtype_{dtype}_nr_args={nr_args}"))

    res = jit_fun1(args)
    self.assertAllClose(args, res, check_dtypes=True)

  def test_tap_jit_large(self):
    arg = jnp.arange(10000, dtype=jnp.int32).reshape((10, 10, 5, -1))
    jax.jit(hcb.id_print)(arg)

  def test_tap_jit_several_together(self):
    arg = jnp.arange(50, dtype=jnp.int32).reshape((10, 5))
    jax.jit(lambda x, y: hcb.id_print((x, y, x * 2.)))(arg, jnp.ones(100, dtype=jnp.int32))

  def test_tap_jit_interleaving(self):
    # Several jit's without data dependencies; they may interfere
    count = 0  # Count tap invocations
    nr_arrays = 5

    def tap_func(arg, _):
      nonlocal count
      assert len(arg) == nr_arrays
      count += 1

    # This is the function that we'll run multiple times
    def func(x, count):
      for i in range(count):
        x = hcb.id_tap(tap_func, [x + i for i in range(nr_arrays)])[-1]
      return x

    x = jnp.array(1, dtype=np.int32)
    res = 0
    for _ in range(10):
      # No dependencies between the jit invocations
      res += jax.jit(lambda x: func(x, 10))(x)
    hcb.barrier_wait()
    self.assertEqual(100, count)

  def test_tap_jit_tap_exception(self):
    if not FLAGS.jax_host_callback_outfeed:
      raise SkipTest("TODO: implement error handling for customcall")
    # Simulate a tap error
    def tap_err(*args, **kwargs):
      raise NotImplementedError

    def func(x):
      x1 = hcb.id_print(x + 1, what="x1", output_stream=testing_stream)
      x2 = hcb.id_tap(tap_err, x1 + 1)
      x3 = hcb.id_print(x2 + 1, what="x3", output_stream=testing_stream)
      return x3

    res = jax.jit(func)(0)  # No error yet
    with self.assertRaises(hcb.CallbackException):
      hcb.barrier_wait()

    # Even though the receiver thread raised, the main thread should still
    # return 3.
    self.assertEqual(3, res)
    # We should have received all others
    assertMultiLineStrippedEqual(self, """
        what: x1
        1
        what: x3
        3""", testing_stream.output)

  def test_tap_while(self):
    """Executing while, even without JIT uses compiled code"""
    y = jnp.ones(5)  # captured const

    def func(x):
      return lax.while_loop(
          lambda c: c[1] < 5,
          lambda c: (y, hcb.id_print(c[1], output_stream=testing_stream) + 1),
          (x, 1))

    func(y)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        1
        2
        3
        4""", testing_stream.output)

  def test_tap_jvp(self):
    jvp_fun1 = lambda x, xt: jax.jvp(fun1, (x,), (xt,))
    res_primals, res_tangents = jvp_fun1(jnp.float32(5.), jnp.float32(0.1))
    self.assertAllClose(100., res_primals, check_dtypes=False)
    self.assertAllClose(4., res_tangents, check_dtypes=False)
    hcb.barrier_wait()
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, """
          transforms: ['jvp'] what: a * 2
          ( 10.00 0.20 )
          transforms: ['jvp'] what: y * 3
          ( 30.00 0.60 )""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          what: a * 2
          10.00
          what: y * 3
          30.00""", testing_stream.output)

  def test_tap_grad_primal_unused(self):
    # The output of id_print is not needed for backwards pass
    def func(x):
      return 2. * hcb.id_print(x * 3., what="x * 3",
                               output_stream=testing_stream)

    grad_func = jax.grad(func)
    arg = jnp.float32(5.)
    jaxpr = str(jax.make_jaxpr(grad_func)(arg))
    # making the Jaxpr does not print anything
    hcb.barrier_wait()

    treedef = tree_util.tree_structure(arg)
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, f"""
        {{ lambda ; a:f32[]. let
            b:f32[] = mul a 3.00
            c:f32[] = outside_call[
              arg_treedef={treedef}
              callback=...
              identity=True
              transforms=()
            ] b
            _:f32[] = mul c 2.00
            d:f32[] = mul 1.00 2.00
            e:f32[] = outside_call[
              arg_treedef={treedef}
              callback=...
              identity=True
              transforms=(('jvp',), ('transpose',))
            ] d
            f:f32[] = mul e 3.00
          in (f,) }}""", jaxpr)
    else:
      assertMultiLineStrippedEqual(self, f"""
        {{ lambda ; a:f32[]. let
            b:f32[] = mul a 3.00
            c:f32[] = outside_call[
              arg_treedef={treedef}
              callback=...
              identity=True
            ] b
            _:f32[] = mul c 2.00
            d:f32[] = mul 1.00 2.00
            e:f32[] = mul d 3.00
          in (e,) }}""", jaxpr)
    assertMultiLineStrippedEqual(self, "", testing_stream.output)
    testing_stream.reset()

    res_grad = grad_func(arg)
    hcb.barrier_wait()

    self.assertAllClose(6., res_grad, check_dtypes=False)
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, """
          what: x * 3
          15.00
          transforms: ['jvp', 'transpose'] what: x * 3
          2.00""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          what: x * 3
          15.00""", testing_stream.output)

  def test_tap_grad_simple(self):
    def func(x):
      y = hcb.id_print(x * 2., what="x * 2", output_stream=testing_stream)
      return x * hcb.id_print(y * 3., what="y * 3",
                              output_stream=testing_stream)

    grad_func = jax.grad(func)

    res_grad = grad_func(jnp.float32(5.))
    self.assertAllClose(2. * 5. * 6., res_grad, check_dtypes=False)
    hcb.barrier_wait()
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, """
          what: x * 2
          10.00
          what: y * 3
          30.00
          transforms: ['jvp', 'transpose'] what: y * 3
          5.00
          transforms: ['jvp', 'transpose'] what: x * 2
          15.00""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          what: x * 2
          10.00
          what: y * 3
          30.00""", testing_stream.output)

  def test_tap_grad_grad(self):
    def func(x):
      y = hcb.id_print(x * 2., what="x * 2", output_stream=testing_stream)
      return x * (y * 3.)

    grad_func = jax.grad(jax.grad(func))
    # making the Jaxpr does not print anything
    _ = jax.make_jaxpr(grad_func)(5.)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, "", testing_stream.output)

    res_grad = grad_func(jnp.float32(5.))

    self.assertAllClose(12., res_grad, check_dtypes=False)
    hcb.barrier_wait()
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, """
          what: x * 2
          10.00
          transforms: ['jvp', 'transpose'] what: x * 2
          15.00
          transforms: ['jvp', 'transpose', 'jvp', 'transpose'] what: x * 2
          2.00
          transforms: ['jvp', 'transpose'] what: x * 2
          3.00""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          what: x * 2
          10.00""", testing_stream.output)

  def test_tap_grad_pytree(self):
    def func(x):
      x4, x5 = hcb.id_print((x * 2., x * 3.), what="pair",
                            result=(x * 4., x * 5.),
                            output_stream=testing_stream)
      return x4 + 2. * x5

    x = jnp.float32(5.)
    grad_func = jax.grad(func)
    print(jax.make_jaxpr(grad_func)(x))
    res_grad = grad_func(x)
    self.assertAllClose(14., res_grad, check_dtypes=False)
    hcb.barrier_wait()
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, """
          what: pair
          ( 10.00 15.00 )
          transforms: ['jvp', 'transpose'] what: pair
          ( 0.00 0.00 )""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          what: pair
          ( 10.00 15.00 )""", testing_stream.output)

  def test_tap_jvp_float0(self):
    def f(x, yint):
      x, yint = hcb.id_tap(lambda arg, _: arg, (x, yint))
      return x * yint

    res = jax.jvp(f, (2., 3), (0.2, np.zeros((), dtypes.float0)))
    self.assertAllClose((6., 0.6), res)

  def test_tap_grad_float0(self):
    def func(x, yint):
      x, yint = hcb.id_print((x, yint), what="pair", output_stream=testing_stream)
      return x * yint

    grad_func = jax.grad(func)

    res_grad = grad_func(jnp.float32(5.), jnp.int32(2))
    self.assertAllClose(2., res_grad, check_dtypes=False)
    hcb.barrier_wait()
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, """
          what: pair
          ( 5.00 2 )
          transforms: ['jvp', 'transpose'] what: pair
          ( 2.00 False )""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          what: pair
          ( 5.00 2 )""", testing_stream.output)

  def test_tap_grad_float0_result(self):
    # https://github.com/google/jax/issues/7340
    # x is a Tuple[f32[2], s32[3]]
    x = (np.array([.7, .8], dtype=np.float32),
         np.array([11, 12, 13], dtype=np.int32))
    def f_jax(x):
      x = hcb.id_print(x, result=x, output_stream=testing_stream)  # result= is important
      return (3. * x[0], x[1])

    def f_jax_vjp(x):
      res, pullback = jax.vjp(f_jax, x)
      g, = pullback((np.ones(x[0].shape, dtype=x[0].dtype),
                     np.zeros(x[1].shape, dtype=dtypes.float0)))
      return g

    g = f_jax_vjp(x)
    self.assertAllClose(np.array([3., 3.], dtype=np.float32), g[0])
    self.assertEqual(dtypes.float0, g[1].dtype)
    hcb.barrier_wait()
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, """
          ( [0.70 0.80] [11 12 13] )
          transforms: ['jvp', 'transpose']
          ( [0.00 0.00] [False False False] )""", testing_stream.output)
    else:
      assertMultiLineStrippedEqual(self, """
          ( [0.70 0.80] [11 12 13] )""", testing_stream.output)

  def test_tap_higher_order_grad_float0_result(self):
    # https://github.com/google/jax/issues/7340
    # x is a Tuple[f32[2], s32[3]]
    x = (np.array([.7, .8], dtype=np.float32),
         np.array([11, 12, 13], dtype=np.int32))
    def f_jax(x):
      x = hcb.id_print(x, result=x, output_stream=testing_stream)  # result= is important
      return (jnp.sin(x[0]), x[1])

    def wrap_vjp(f, args, res_f_of_args):
      # Given a function "f" and "args" return the f_vjp and args_vjp
      def make_ct(res):
        res_dtype = np.result_type(res)
        if res_dtype == dtypes.float0:
          return res
        ct_dtype = core.primal_dtype_to_tangent_dtype(res_dtype)
        return np.ones(np.shape(res), dtype=ct_dtype)
      cts = tree_util.tree_map(make_ct, res_f_of_args)
      def f_vjp(args, cts):
        res, pullback = jax.vjp(f, *args)
        return pullback(cts)
      return (f_vjp, (args, cts))

    res = f_jax(x)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        ( [0.70 0.80] [11 12 13] )""", testing_stream.output)
    testing_stream.reset()

    # 1st order
    f_jax_vjp1, args_vjp1 = wrap_vjp(f_jax, (x,), res)
    res_vjp1 = f_jax_vjp1(*args_vjp1)
    hcb.barrier_wait()
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiLineStrippedEqual(self, """
          ( [0.70 0.80] [11 12 13] )
          transforms: ['jvp', 'transpose']
          ( [0.00 0.00] [False False False] )""", testing_stream.output)
    else:
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
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        transforms: [('batch', {'batch_dims': (0,)})] what: a * 2
        [ 8.00 10.00]
        transforms: [('batch', {'batch_dims': (0,)})] what: y * 3
        [24.00 30.00]""", testing_stream.output)

  def test_tap_vmap_not_batched(self):
    x = 3.

    def func(y):
      # x is not mapped, y is mapped
      _, y = hcb.id_print((x, y), output_stream=testing_stream)
      return x + y

    vmap_func = jax.vmap(func)
    vargs = jnp.array([jnp.float32(4.), jnp.float32(5.)])
    _ = vmap_func(vargs)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
      transforms: [('batch', {'batch_dims': (None, 0)})]
      ( 3.00 [4.00 5.00] )""", testing_stream.output)

  def test_tap_vmap_vmap(self):
    # A 2D tensor with x[i, j] = i + j using 2 vmap
    def sum(x, y):
      return hcb.id_print(x + y, output_stream=testing_stream)

    def sum_rows(xv, y):
      return jax.vmap(sum, in_axes=(0, None))(xv, y)

    def sum_all(xv, yv):
      return jax.vmap(sum_rows, in_axes=(None, 0))(xv, yv)

    xv = jnp.arange(5, dtype=np.int32)
    yv = jnp.arange(3, dtype=np.int32)
    # assertMultiLineStrippedEqual(self, "", str(jax.make_jaxpr(sum_all)(xv, yv)))
    _ = sum_all(xv, yv)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        transforms: [('batch', {'batch_dims': (0,)}), ('batch', {'batch_dims': (0,)})]
        [[0 1 2 3 4]
        [1 2 3 4 5]
        [2 3 4 5 6]]""", testing_stream.output)

  def test_tap_vmap_while(self):
    """Vmap of while."""

    def func(x):
      # like max(x, 2)
      x1 = hcb.id_print(x, where="before:x", output_stream=testing_stream)
      x2 = lax.while_loop(
          lambda x: x < 2, lambda x: hcb.id_print(
              x + 1, where="body:x+1", output_stream=testing_stream), x1)
      res = hcb.id_print(x2, where="after:x", output_stream=testing_stream)
      return res

    inputs = np.arange(5, dtype=np.int32)
    self.assertAllClose(
        np.array([2, 2, 2, 3, 4]),
        jax.jit(jax.vmap(func))(inputs),
        check_dtypes=False)
    hcb.barrier_wait()
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

  def test_tap_vmap_while_tap_cond(self):
    """Vmap of while, with a tap in the conditional."""

    def func(x):
      # like max(x, 2)
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = lax.while_loop(lambda x: hcb.id_print(x < 2, where="w_c",
                                                 output_stream=testing_stream),
                          lambda x: hcb.id_print(x + 1, where="w_b",
                                                 output_stream=testing_stream),
                          x1)
      res = hcb.id_print(x2, where="3", output_stream=testing_stream)
      return res

    inputs = np.arange(5, dtype=np.int32)
    res = jax.jit(jax.vmap(func))(inputs)
    hcb.barrier_wait()
    self.assertAllClose(np.array([2, 2, 2, 3, 4]), res, check_dtypes=False)
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

  def test_tap_transforms_old_doc(self):
    if not FLAGS.jax_host_callback_ad_transforms:
      raise unittest.SkipTest("disabled for new behavior")

    # Examples from the documentation
    def power3(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      _, y = hcb.id_print((x, y), what="x,x^2", output_stream=testing_stream)
      return y * x

    print(f"impl = {power3(3.)}")
    hcb.barrier_wait()
    expected = """
       what: x,x^2
      ( 3. 9. )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"vmap = {jax.vmap(power3)(np.arange(3.))}")
    hcb.barrier_wait()
    expected = """
      transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
      ( [0. 1. 2.] [0. 1. 4.] )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"jvp = {jax.jvp(power3, (3.,), (0.1,))}")
    hcb.barrier_wait()
    expected = """
      transforms: ['jvp'] what: x,x^2
      ( ( 3. 9. ) ( 0.1 0.6 ) )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"grad = {jax.grad(power3)(3.)}")
    hcb.barrier_wait()
    expected = """
      what: x,x^2
      ( 3. 9. )
      transforms: ['jvp', 'transpose'] what: x,x^2
      ( 0. 3. )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"vmap o grad {jax.vmap(jax.grad(power3))(np.array([2., 3.]))}")
    hcb.barrier_wait()
    expected = """
      transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
      ( [2. 3.] [4. 9.] )
      transforms: ['jvp', 'transpose', ('batch', {'batch_dims': (None, 0)})] what: x,x^2
      ( 0. [2. 3.] )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_tap_transforms_doc(self):
    # Examples from the documentation
    if FLAGS.jax_host_callback_ad_transforms:
      raise unittest.SkipTest("disabled for old behavior")
    def power3(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb.id_print((x, y), what="x,x^2", output_stream=testing_stream)
      return y * x

    print(f"impl = {power3(3.)}")
    hcb.barrier_wait()
    expected = """
        what: x,x^2
       ( 3. 9. )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"jvp = {jax.jvp(power3, (3.,), (0.1,))}")
    hcb.barrier_wait()
    expected = """
         what: x,x^2
         ( 3. 9. )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    @jax.custom_jvp
    def print_tangents(arg):
      return None

    @print_tangents.defjvp
    def print_tangents_jvp(primals, tangents):
      arg_dot, = tangents
      hcb.id_print(arg_dot, what="tangents", output_stream=testing_stream)
      return primals, tangents

    def power3_with_tangents(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb.id_print((x, y), what="x,x^2", output_stream=testing_stream)
      print_tangents((x, y))
      return y * x

    print(f"jvp = {jax.jvp(power3_with_tangents, (3.,), (0.1,))}")
    hcb.barrier_wait()
    expected = """
      what: x,x^2
      ( 3. 9. )
      what: tangents
      ( 0.1 0.6 )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"grad = {jax.grad(power3)(3.)}")
    hcb.barrier_wait()
    # Only the primals by default
    expected = """
       what: x,x^2
       ( 3. 9. )"""
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
      hcb.id_print(ct_b, what="cotangents", output_stream=testing_stream)
      return ct_b,

    print_cotangents.defvjp(print_cotangents_fwd, print_cotangents_bwd)

    def power3_with_cotangents(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb.id_print((x, y), what="x,x^2", output_stream=testing_stream)
      # Must use the output of print_cotangents
      (x1, y1) = print_cotangents((x, y))
      return y1 * x1

    print(f"grad = {jax.grad(power3_with_cotangents)(3.)}")
    hcb.barrier_wait()
    expected = """
      what: x,x^2
      ( 3. 9. )
      what: cotangents
      ( 9. 3. )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    # TODO: grad of grad

    print(f"vmap = {jax.vmap(power3)(np.array([2., 3.]))}")
    hcb.barrier_wait()
    expected = """
       transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
       ( [2. 3.] [4. 9.] )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"vmap o grad {jax.vmap(jax.grad(power3))(np.array([2., 3.]))}")
    hcb.barrier_wait()
    expected = """
       transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
       ( [2. 3.] [4. 9.] )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"vmap o grad {jax.vmap(jax.grad(power3_with_cotangents))(np.array([2., 3.]))}")
    hcb.barrier_wait()
    expected = """
      transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
      ( [2. 3.] [4. 9.] )
      transforms: [('batch', {'batch_dims': (0, 0)})] what: cotangents
      ( [4. 9.] [2. 3.] )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print(f"grad o remat = {jax.grad(lambda x: power3(ad_checkpoint.checkpoint(power3)(x)))(3.)}")
    hcb.barrier_wait()
    expected = """
      what: x,x^2
      ( 3. 9. )
      what: x,x^2
      ( 27. 729. )
      what: x,x^2
      ( 3. 9. )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

  def test_tap_pmap(self):
    if len(local_devices()) < 2:
      raise SkipTest("test requires at least 2 devices")

    def power3(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      _, y = hcb.id_print((x, y),
                          what="x,x^2",
                          output_stream=testing_stream,
                          tap_with_device=True)
      return y * x

    pmap_power3 = jax.pmap(power3, devices=local_devices())
    xv = np.array([3, 4], dtype=np.int32)
    res = pmap_power3(xv)
    hcb.barrier_wait()
    self.assertAllClose(xv * xv * xv, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(
        self, """
      device: cpu:0 what: x,x^2
      ( 3 9 )
      device: cpu:1 what: x,x^2
      ( 4 16 )""")

  def test_tap_pmap_vmap(self):
    # A matrix M[ij] = i * 10 + j
    nr_devices = len(local_devices())
    shape = (nr_devices, 3)
    matrix = np.fromfunction(lambda i, j: 10. * i + j, shape,
                             dtype=np.int32)

    def fun1(x, do_print=False):  # x: i32
      return maybe_print(do_print, x * 2, "x * 2", tap_with_device=True)

    pmap_vmap_fun1 = jax.pmap(
        jax.vmap(partial(fun1, do_print=True)), devices=local_devices())

    res = pmap_vmap_fun1(matrix)
    hcb.barrier_wait()
    expected_res = jax.pmap(
        jax.vmap(partial(fun1, do_print=False)), devices=local_devices())(
            matrix)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [0.00 2.00 4.00]
        device: cpu:1 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [20.00 22.00 24.00]""")

  def test_tap_pmap_pmap_vmap(self):
    # A matrix M[ijk] = i * 100 + j * 10 + k
    nr_devices = len(local_devices())
    if nr_devices % 2 != 0:
      raise SkipTest("test works only on even number of devices")

    shape = (2, nr_devices // 2, 3)
    matrix = np.fromfunction(lambda i, j, k: 100. * i + 10. * j + k, shape,
                             dtype=np.float32)

    def fun1(x, do_print=False):  # x: f32
      y = maybe_print(do_print, x * 2., "x * 2", tap_with_device=True)
      return y ** 2

    pmap_fun1 = jax.pmap(
        jax.pmap(jax.vmap(partial(fun1, do_print=True))),
        devices=local_devices())
    res = pmap_fun1(matrix)
    hcb.barrier_wait()
    expected_res = jax.pmap(
        jax.pmap(jax.vmap(partial(fun1, do_print=False))),
        devices=local_devices())(
            matrix)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [0.00 2.00 4.00]
        device: cpu:1 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [200.00 202.00 204.00]""")

  @ignore_jit_of_pmap_warning()
  def test_tap_pmap_pmap_extra(self):
    """pmap of a pmap surrounded by extra code."""
    # A matrix M[ij] = i * 10 + j
    nr_devices = len(local_devices())
    if nr_devices != 2:
      raise SkipTest("test works only on 2 devices")
    shape = (2, 1, 3)
    matrix = np.fromfunction(lambda i, j, k: 100. * i + 10. * j + k, shape,
                             dtype=np.float32)

    def fun(xv, do_print=False):
      # This will be printed on all devices, with shape [1, 3]
      xv = maybe_print(do_print, xv + 1., "before", tap_with_device=True)
      res = jax.pmap(lambda x: maybe_print(do_print, x * 2., "inside", tap_with_device=True))(xv)
      # This will be printed on all devices, with shape [1, 3]
      return maybe_print(do_print, res + 1., "after", tap_with_device=True)

    res = jax.pmap(partial(fun, do_print=True))(matrix)
    self.assertAllClose(fun(matrix, do_print=False), res, check_dtypes=False)
    hcb.barrier_wait()
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
      device: cpu:0 what: before
      [[1.00 2.00 3.00]]
      device: cpu:0 what: inside
      [2.00 4.00 6.00]
      device: cpu:0 what: after
      [[3.00 5.00 7.00]]
      device: cpu:1 what: before
      [[101.00 102.00 103.00]]
      device: cpu:1 what: inside
      [202.00 204.00 206.00]
      device: cpu:1 what: after
      [[203.00 205.00 207.00]]""")

  def test_tap_jvp_pmap_vmap(self):
    # A matrix M[ijk] = i * 100 + j * 10 * k
    nr_devices = len(local_devices())
    shape = (nr_devices, 2, 3)
    matrix = np.fromfunction(lambda i, j, k: 100. * i + 10. * j + k, shape,
                             dtype=np.float32)

    def fun(xv, do_print=False):
      # x: f32[3]
      return jax.jvp(jax.pmap(jax.vmap(lambda x: maybe_print(do_print, x * 2., "x * 2", tap_with_device=True))),
                     (xv,), (.1 * jnp.ones_like(xv),))

    res = fun(matrix, do_print=True)
    hcb.barrier_wait()
    expected_res = fun(matrix, do_print=False)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    # Device 0 will get to execute jax.jvp(jax.vmap(...)) for matrix[0, :, :]
    if FLAGS.jax_host_callback_ad_transforms:
      assertMultiDeviceOutputEqual(self, """
        device: cpu:0 transforms: [('batch', {'batch_dims': (0,)}), 'jvp'] what: x * 2
        ( [[ 0.00  2.00  4.00]
           [20.00 22.00 24.00]] [[0.20 0.20 0.20]
           [0.20 0.20 0.20]] )
        device: cpu:1 transforms: [('batch', {'batch_dims': (0,)}), 'jvp'] what: x * 2
        ( [[200.00 202.00 204.00]
           [220.00 222.00 224.00]] [[0.20 0.20 0.20]
           [0.20 0.20 0.20]] )""")
    else:
      assertMultiDeviceOutputEqual(self, """
        device: cpu:0 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [[ 0.00  2.00  4.00]
         [20.00 22.00 24.00]]
        device: cpu:1 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [[200.00 202.00 204.00]
         [220.00 222.00 224.00]]""")

  def test_tap_vmap_pmap(self):
    # A matrix M[ijk] = i * 100 + j * 10 * k
    nr_devices = len(local_devices())
    shape = (2, nr_devices, 3)
    matrix = np.fromfunction(lambda i, j, k: 100. * i + 10. * j + k, shape,
                             dtype=np.float32)

    def fun(xv, do_print=False):
      # x: f32[3]
      return jax.vmap(jax.pmap(lambda x: maybe_print(do_print, x * 2., "x * 2", tap_with_device=True)))(xv)

    res = fun(matrix, do_print=True)
    hcb.barrier_wait()
    expected_res = fun(matrix, do_print=False)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    # Device 0 will get to execute jax.jvp(jax.vmap(...)) for matrix[:, 0, :]
    assertMultiDeviceOutputEqual(self, """
      device: cpu:0 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
      [[  0.00   2.00   4.00]
       [200.00 202.00 204.00]]
      device: cpu:1 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
      [[ 20.00  22.00  24.00]
       [220.00 222.00 224.00]]""")

  @ignore_jit_of_pmap_warning()
  def test_tap_jit_pmap_extra(self):
    """jit of a pmap surrounded by extra code."""
    # A matrix M[ij] = i * 10 + j
    nr_devices = len(local_devices())
    assert nr_devices in (1, 2)
    shape = (nr_devices, 3)
    matrix = np.fromfunction(lambda i, j: 10. * i + j, shape,
                             dtype=np.float32)

    def fun(xv, do_print=False):
      # This will be printed on all devices with shape (nr_devices, 3)
      xv = maybe_print(do_print, xv + 1., "before", tap_with_device=True)
      res = jax.pmap(lambda x: maybe_print(do_print, x * 2., "inside", tap_with_device=True))(xv)
      # This will be printed on all devices with shape (nr_devices, 3)
      return maybe_print(do_print, res + 1., "after", tap_with_device=True)

    res = jax.jit(partial(fun, do_print=True))(matrix)
    self.assertAllClose(fun(matrix, do_print=False), res, check_dtypes=False)
    hcb.barrier_wait()
    if len(local_devices()) == 2:
      assertMultiDeviceOutputEqual(self, """
        device: cpu:0 what: before
        [[ 1.00  2.00  3.00]
         [11.00 12.00 13.00]]
        device: cpu:0 what: inside
        [2.00 4.00 6.00]
        device: cpu:0 what: after
        [[ 3.00  5.00  7.00]
         [23.00 25.00 27.00]]
        device: cpu:1 what: before
        [[ 1.00  2.00  3.00]
         [11.00 12.00 13.00]]
        device: cpu:1 what: inside
        [22.00 24.00 26.00]
        device: cpu:1 what: after
        [[ 3.00  5.00  7.00]
         [23.00 25.00 27.00]]""")
    else:
      assert len(local_devices()) == 1
      assertMultiDeviceOutputEqual(self, """
        device: cpu:0 what: before
        [[1.00 2.00 3.00]]
        device: cpu:0 what: inside
        [2.00 4.00 6.00]
        device: cpu:0 what: after
        [[3.00 5.00 7.00]]""")

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
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        TBD""", testing_stream.output)

  @jtu.skip_on_devices("cpu", "gpu")
  # TODO(necula): file XLA:GPU bug for the 'Sharding' CustomCall
  def test_tap_pjit(self):
    devices = np.array(local_devices())
    nr_devices = len(devices)
    if nr_devices < 2:
      raise SkipTest("test requires at least 2 devices")

    print(f"test_tap_pjit is running on devices {devices}.")
    # x: i32[D, 3] = [[0, 1, 2], [10, 11, 12], ...]
    # y: i32[3, 4]
    x = jnp.arange(100, dtype=jnp.int32).reshape((10, 10))[:nr_devices, :3]
    y = jnp.ones((3, 4), np.int32)

    @partial(jax.named_call, name="fun1")  # for xprof debugging
    def fun1(x, do_print=False):
      z = jnp.dot(x, y)
      return maybe_print(do_print, z, "z", tap_with_device=True)

    res0 = fun1(x, do_print=False)
    pjit_fun1 = pjit.pjit(
        partial(fun1, do_print=True),
        in_axis_resources=(P("d"),),
        out_axis_resources=P("d"))

    with maps.mesh(devices, ["d"]):
      # Print the internal IR
      helper_log_ir(
          f"{self._testMethodName}.pjit",
          pjit_fun1,
          x,
          num_partitions=nr_devices)
      res = pjit_fun1(x)

    self.assertAllClose(res0, res)
    hcb.barrier_wait("before check")

    # Assertion text is for 2 devices (also works for 1 device)
    # Note that a single call is made.
    assertMultiDeviceOutputEqual(
        self, """
       device: cpu:0 what: z
       [[ 3  3  3  3]
        [33 33 33 33]]""")

  def test_tap_scan_custom_jvp(self):
    """custom JVP, inside scan.
    This exercises the custom_jvp_call_jaxpr primitives."""

    @jax.custom_jvp
    def f(x):
      return x * hcb.id_print(x, output_stream=testing_stream, what="x")

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * hcb.id_print(x_dot, output_stream=testing_stream, what="x_dot")
      return primal_out, tangent_out

    def g(x):
      # Sum f(x_i)
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((2,), 0.7)
    self.assertAllClose(0.7 * 0.7 * 2, g(arg))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        what: x
        0.7
        what: x
        0.7""", testing_stream.output)
    testing_stream.reset()

    self.assertAllClose(np.array([2.1, 2.1]), jax.grad(g)(arg), check_dtypes=False)
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        what: x
        0.7
        what: x
        0.7
        transforms: ['transpose'] what: x_dot
        2.1
        transforms: ['transpose'] what: x_dot
        2.1""", testing_stream.output)

  def test_tap_scan_custom_vjp(self):
    """custom VJP, inside scan.
    This exercises the custom_vjp_call_jaxpr primitives."""

    @jax.custom_vjp
    def f(x):
      return x * hcb.id_print(x, output_stream=testing_stream, what="x")

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x

    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * hcb.id_print(ct_b, output_stream=testing_stream, what="ct_b"),

    f.defvjp(f_fwd, f_bwd)

    def g(x):
      # Sum f(x_i)
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((2,), 0.7)

    self.assertAllClose(0.7 * 0.7 * 2, g(arg))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        what: x
        0.7
        what: x
        0.7""", testing_stream.output)
    testing_stream.reset()

    self.assertAllClose(np.array([2.1, 2.1]), jax.grad(g)(arg), check_dtypes=False)
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        what: x
        0.7
        what: x
        0.7
        what: ct_b
        1.
        what: ct_b
        1.""", testing_stream.output)

  def test_tap_mask(self):

    @partial(jax.mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      three_x = hcb.id_print((x, 2 * x), result=3 * x, what="x",
                             output_stream=testing_stream)
      return jnp.sum(three_x)

    x = np.arange(5.)

    self.assertAllClose(9., padded_sum([x], dict(n=3)))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        transforms: [('mask', {'logical_shapes': 5})] what: x
        ( ( [0. 1. 2. 3. 4.] [0. 2. 4. 6. 8.] ) ( ( 3 ) ( 3 ) ) )""",
        testing_stream.output)
    testing_stream.reset()

    # With VMAP
    xv = np.arange(10.).reshape((2, 5))  # logical_shape = 5
    self.assertAllClose(
        np.array([9., 78.]),
        # batch_size = 2, n=3 and 4 for the two elements
        jax.vmap(padded_sum)([xv],
                             dict(n=np.array([3., 4.]))))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        transforms: [('mask', {'logical_shapes': 5}), ('batch', {'batch_dims': (0, 0, 0, 0)})] what: x
        ( ( [[0. 1. 2. 3. 4.]
             [5. 6. 7. 8. 9.]]
            [[ 0.  2.  4.  6.  8.]
             [10. 12. 14. 16. 18.]] )
          ( ( [3. 4.] ) ( [3. 4.] ) ) )""", testing_stream.output)
    testing_stream.reset()

    # With JVP
    self.assertAllClose((9., 0.9),
                        jax.jvp(lambda arg: padded_sum([arg], dict(n=3)),
                                (x,), (x * 0.1,)))
    hcb.barrier_wait()
    if FLAGS.jax_host_callback_ad_transforms:
      self.assertMultiLineStrippedEqual("""
          transforms: [('mask', {'logical_shapes': 5}), 'jvp'] what: x
          ( ( ( [0. 1. 2. 3. 4.] [0. 2. 4. 6. 8.] ) ( ( 3 ) ( 3 ) ) )
            ( ( [0.  0.1 0.2 0.3 0.4] [0.  0.2 0.4 0.6 0.8] ) ( ( False ) ( False ) ) ) )""",
            testing_stream.output)
    else:
      self.assertMultiLineStrippedEqual("""
          transforms: [('mask', {'logical_shapes': 5})] what: x
          ( ( [0. 1. 2. 3. 4.] [0. 2. 4. 6. 8.] ) ( ( 3 ) ( 3 ) ) )""",
            testing_stream.output)
    testing_stream.reset()

    # Now with JIT
    self.assertAllClose(9., jax.jit(padded_sum)([x], dict(n=3)))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
      transforms: [('mask', {'logical_shapes': 5})] what: x
      ( ( [0. 1. 2. 3. 4.] [0. 2. 4. 6. 8.] ) ( ( 3 ) ( 3 ) ) )""",
      testing_stream.output)

  def test_tap_callback_delay(self):
    hcb.callback_extra = lambda dev: time.sleep(1)

    def func(x):
      for i in range(5):
        x = hcb.id_print(x * i, what="x times i")
      return x

    jax.jit(func)(np.arange(6, dtype=np.float32).reshape((2, 3)))

  def test_tap_callback_delay_barrier(self):
    hcb.callback_extra = lambda dev: time.sleep(2)

    def func(x):
      for i in range(1, 4):
        x = hcb.id_print(x * i, what=f"x times {i}", output_stream=testing_stream)
      return x

    jax.jit(func)(np.arange(6, dtype=np.float32).reshape((2, 3)))
    # Wait for the results
    hcb.barrier_wait("first")
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
    hcb.barrier_wait("second")
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
          [xops.Constant(comp, np.zeros((2, 3), dtype=np.float32))])

  def test_tap_error_different_shapes(self):
    """Try to register different shapes for the same consumer ID."""
    if not hcb._use_outfeed(jtu.device_under_test()):
      raise SkipTest("test works only for outfeed")
    comp = xla_client.XlaBuilder(self._testMethodName)
    token = hcb.xops.CreateToken(comp)
    hcb._initialize_outfeed_receiver()  # Needed if this is the sole test
    hcb._callback_handler_data.receiver.add_outfeed(
        comp, token, 123,
        [xops.Constant(comp, np.zeros((2, 3), dtype=np.float32))])
    with self.assertRaisesRegex(
        RuntimeError, ".*does not match previous shape element_type.*"):
      hcb._callback_handler_data.receiver.add_outfeed(
          comp, token, 123,
          [xops.Constant(comp, np.zeros((2, 3), dtype=np.int32))])
    with self.assertRaisesRegex(
        RuntimeError, ".*does not match previous shape element_type.*"):
      hcb._callback_handler_data.receiver.add_outfeed(
          comp, token, 123,
          [xops.Constant(comp, np.zeros((2,), dtype=np.float32))])

  def test_tap_id_tap_removed_kwargs(self):
    def func(x, transforms, y):
      pass

    with self.assertRaisesRegex(TypeError, r"Support for \*\*kwargs in ``id_tap``"):
      hcb.id_tap(func, 1, y=2)

  def test_tap_odeint(self):
    # TODO: find a smaller repro for bug #4015
    # Seems to be xla_call(scan(xla_call)), all under grad.
    from jax.experimental.ode import odeint

    def f(x, t, k):
      x = hcb.id_print(x)
      return -k * x

    def loss(k=1.0):
      t = jnp.linspace(0, 0.001, num=2)
      xs = odeint(f, 1.0, t, k)
      return xs[-1]

    jax.grad(loss)(1.0)  # should not fail

  def test_tap_remat_0(self):
    def f(i, k):
      x = hcb.id_print(k + i, output_stream=testing_stream)
      return k * x

    def loss(k):
      return lax.fori_loop(0, 2, jax.remat(f), k)

    print(loss(3))
    hcb.barrier_wait()
    expected = """
      3
      10"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(testcase_name=f"_use_remat={use_remat}_{grad_func}_use_result={use_result}",
               use_result=use_result, use_remat=use_remat, grad_func=grad_func)
          for use_result in [True, False]
          for grad_func in ["grad", "value_and_grad"]
          for use_remat in ["old", "new", "none"]))
  def test_tap_remat(self, use_result=False, grad_func="grad", use_remat="new"):
    def f(x):
      id_print_result = hcb.id_print(x, output_stream=testing_stream)
      if use_result:
        x = id_print_result
      return 3. * x
    grad_f = jax.grad if grad_func == "grad" else jax.value_and_grad
    if use_remat == "old":
      trans_f = jax.remat(f)
    elif use_remat == "new":
      trans_f = ad_checkpoint.checkpoint(f)
    else:
      assert use_remat == "none"
      trans_f = f
    print(jax.make_jaxpr(grad_f(trans_f))(2.))
    grad_f(trans_f)(2.)

    hcb.barrier_wait()

    if use_remat == "none":
      if use_result:
        if FLAGS.jax_host_callback_ad_transforms:
          expected = """
            2.
            transforms: ['jvp', 'transpose']
            3."""
        else:
          # GOOD: whether or not we use_result, in absence of
          # jax_host_callback_ad_transforms we get the same callback.
          expected = "2."
      else:
        expected = "2."
    else:  # use_remat
      if use_result:
        if FLAGS.jax_host_callback_ad_transforms:
          expected = """
            2.
            2.
            transforms: ['jvp', 'transpose']
            3."""
        else:
          expected = """
            2.
            2."""
      else:
        if use_remat == "old":
          # TODO: we should see two callbacks
          expected = ""
        else:
          # Good: we see two callbacks, whether or not we use the result.
          expected = """
            2.
            2."""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_tap_named_call(self):
    def tap_scalar(init, do_print=False):
      @partial(jax.named_call, name="step")
      def step(acc, step_nr):
        acc = acc + step_nr
        maybe_print(do_print, step_nr, what="step_nr")
        return acc, None

      return lax.scan(step, init, np.arange(2))

    self.assertAllClose(tap_scalar(3., do_print=False), tap_scalar(3., do_print=True))
    hcb.barrier_wait()
    expected = """
      what: step_nr
      0
      what: step_nr
      1"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)


class HostCallbackCallTest(jtu.JaxTestCase):
  """Tests for hcb.call"""

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() == "gpu" and jax.device_count() > 1:
      raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")

    testing_stream.reset()
    testing_stream._test_method_name = self._testMethodName

  def tearDown(self) -> None:
    hcb.barrier_wait("HostCallbackCallTest.tearDown")
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(testcase_name=f"_{np.dtype(dtype).name}", dtype=dtype)
          for dtype in jtu.dtypes.all
          if dtype != np.bool_))
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

  def test_call_empty_arg(self):
    """Call with empty array."""
    result = np.ones((2,), dtype=np.float32)
    def f_outside(_):
      return result
    def fun(x):
      return x + hcb.call(f_outside, (),
                          result_shape=jax.ShapeDtypeStruct(result.shape, result.dtype))
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
    hcb.barrier_wait()
    expected = """
        Call outside([3. 4.])
          = [3. 4.]"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_call_cond(self):
    def f_outside(args):
      x, y = args
      return x * y

    def loop(x, use_outside=True):
      def body(i, acc):
        return lax.cond(i % 2 == 1,
                        lambda _: (hcb.call(f_outside, (acc, i),
                                            result_shape=acc)
                                   if use_outside else f_outside((acc, i))),
                        lambda _: acc,
                        None)

      return lax.fori_loop(0, 18, body, x)

    res_inside = loop(1.2, use_outside=False)
    self.assertAllClose(res_inside, jax.jit(loop)(1.2))

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

  def test_call_with_device(self):
    def callback_func(x, device=None):
      testing_stream.write(f"device: {device}\n Called with {x}")
      return x

    def func(x):
      return hcb.call(callback_func, x,
                      result_shape=x,
                      call_with_device=True)

    self.assertEqual(3., func(3.))
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0
         Called with 3.00""")

  def test_call_pmap(self):
    # Works for 1 or 2 devices
    def callback_func(x, device=None):
      testing_stream.write(f"device: {device}\n Called with {x}")
      return x * np.array(3, np.int32)

    def fun(x):  # x: i32
      return hcb.call(callback_func, x * 2,
                      result_shape=x,
                      call_with_device=True)

    xv = jnp.arange(len(local_devices()), dtype=jnp.int32)
    res = jax.pmap(fun)(xv)
    self.assertAllClose(jax.pmap(lambda x: x * 6)(xv), res)
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0
         Called with 0
        device: cpu:1
         Called with 2""")

  def test_call_vmap(self):
    def f_outside(x): return x

    def fun(x):
      return hcb.call(f_outside, x, result_shape=x)

    with self.assertRaisesRegex(NotImplementedError,
                                "batching rules are implemented only for id_tap, not for call"):
      jax.vmap(fun)(np.ones((2, 3)))

  @jtu.skip_on_devices("cpu", "gpu")
  # TODO(necula): file XLA:GPU bug for the 'Sharding' CustomCall
  def test_call_pjit(self):
    devices = np.array(local_devices())
    nr_devices = len(devices)
    if nr_devices < 2:
      raise SkipTest("test requires at least 2 devices")

    print(f"test_call_pjit is running on devices {devices}.")
    # x: i32[D, 3] = [[0, 1, 2], [10, 11, 12], ...]
    # y: i32[3, 4]
    x = jnp.arange(100, dtype=jnp.int32).reshape((10, 10))[:nr_devices, :3]
    y = jnp.ones((3, 4), np.int32)

    def callback_x5_func(x, device=None):
      testing_stream.write(f"device: {device}\n Called with {x}")
      return x * np.array(5, np.int32)

    def fun(x):
      xy = jnp.dot(x, y)
      return hcb.call(
          callback_x5_func, xy, result_shape=xy, call_with_device=True)

    pjit_fun = pjit.pjit(
        fun, in_axis_resources=(P("d"),), out_axis_resources=P("d"))
    with maps.mesh(devices, ["d"]):
      # Print the internal IR
      helper_log_ir(
          f"{self._testMethodName}.pjit",
          pjit_fun,
          x,
          num_partitions=nr_devices)

      res = pjit_fun(x)

    expected_res = jnp.dot(x, y) * np.array(5, np.int32)
    self.assertAllClose(expected_res, res, check_dtypes=False)

    hcb.barrier_wait("before assertion")
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(
        self, """
        device: cpu:0
         Called with [[ 3  3  3  3]
         [33 33 33 33]]""")

  def test_call_error_bad_result_shape(self):
    with self.assertRaisesRegex(
        ValueError,
        "The values must be either numeric scalars, or must have 'shape' and 'dtype' attributes"):
      hcb.call(lambda x: x, 3., result_shape="string")

    with self.assertRaisesRegex(
        ValueError,
        "The values must be either numeric scalars, or must have 'shape' and 'dtype' attributes"):
      hcb.call(lambda x: x, 3., result_shape=lambda x: x)
      hcb.barrier_wait("wait for error")

  def helper_check_callback_errors(self, thunk: Callable,
                                   expected_exc_txt: str):
    """Calls thunk() and checks for expected exceptions.
    """
    if jtu.device_under_test() == "cpu":
      # On CPU the runtime crashes, and the tests are all aborted
      raise SkipTest("TODO: CPU runtime crashes on unexpected infeed")
    elif jtu.device_under_test() == "gpu":
      # On GPU we get a nice error back to Python
      with self.assertRaisesRegex(
          RuntimeError,
          "RET_CHECK failure .* Mismatch between infeed source buffer shape s8.12345."):
        thunk()
    elif jtu.device_under_test() == "tpu":
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
      hcb.barrier_wait("Waiting for error")

  def test_call_error_callback_throws_exception(self):
    def f_outside(x):
      raise ValueError("user exception")
    def fun(x):
      return hcb.call(f_outside, x, result_shape=x)

    self.helper_check_callback_errors(lambda: fun(3.),
                                      "ValueError: user exception")

  def test_call_error_callback_returns_unexpected_shape(self):
    def fun(x):
      return hcb.call(lambda x: (x, x), x, result_shape=x)

    self.helper_check_callback_errors(lambda: fun(3.),
                                      "Callback func .* should have returned a result with pytree")

  def test_call_error_then_compute(self):
    # Continue computation on device after error
    def f_outside(x):
      raise ValueError("user exception")
    def fun(x):
      x1 = hcb.call(f_outside, x, result_shape=x)
      return x1
    arg = np.arange(3, dtype=np.int32)
    self.helper_check_callback_errors(lambda: self.assertAllClose(arg, fun(arg)),
                                      "ValueError: user exception")


def call_jax_other_device(jax_outside_fun, arg, *, device):
  """Calls a JAX function on a specific device with simple support for reverse AD.

  Functions whose name starts with "jax_outside" are called on another device,
  by way of hcb.call.
  """

  def run_jax_outside_fun(arg):
    return jax.jit(jax_outside_fun)(jax.device_put(arg, device))

  @jax.custom_vjp
  def make_call(arg):
    return hcb.call(run_jax_outside_fun, arg,
                    result_shape=jax.eval_shape(jax_outside_fun, arg))

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
    if jtu.device_under_test() == "gpu" and jax.device_count() > 1:
      raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")

    if jtu.device_under_test() != "cpu":
      assert jax.devices("cpu")
      self.outside_device = jax.devices("cpu")[0]
    else:
      if len(jax.devices("cpu")) == 1:
        raise SkipTest("Test needs at least two devices. On CPU use XLA_FLAGS=--xla_force_host_platform_device_count=2")
      self.outside_device = jax.devices("cpu")[1]
    super().setUp()

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


class OutfeedRewriterTest(jtu.JaxTestCase):

  def setUp(self):
    if jtu.device_under_test() == "gpu" and jax.device_count() > 1:
      raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")
    super().setUp()

  def assertRewrite(self, expected: str, func: Callable, args: Sequence,
                    has_input_token=True, has_output_token=True):
    """Check that the rewrite of func(*args) matches expected."""
    jaxpr = jax.make_jaxpr(func)(*args)
    rewritten = hcb._rewrite_closed_jaxpr(jaxpr,  # noqa: F841
                                          has_input_token, has_output_token)
    # Since it is somewhat annoying to update the Jaxpr assertions when we change
    # the Jaxpr printing, we do not check these by default. It is recommended that
    # before making changes to the code generation and Jaxpr rewriting, turn on
    # the checking, update the expected Jaxpr, and then make the changes.
    # assertMultiLineStrippedEqual(self, expected, str(rewritten))
    del rewritten

  def test_no_outfeed(self):
    self.assertRewrite("""
        { lambda  ; a.
          let b = mul a a
              c = add a b
          in (c,) }""", lambda x: x + x * x, [0], has_input_token=False,
                       has_output_token=False)
    self.assertRewrite("""
        { lambda  ; a d e.
          let b = mul a a
              c = add a b
          in (c,) }""", lambda x: x + x * x, [0], has_output_token=False)
    self.assertRewrite("""
        { lambda  ; a d e.
          let b = mul a a
              c = add a b
          in (c, d, e) }""", lambda x: x + x * x, [0])

  def test_simple_outfeed(self):
    self.assertRewrite("""
        { lambda  ; a d e.
          let b = add a a
              c f g = outside_call[ arg_treedef=*
                                    callback=...
                                    has_token=True
                                    identity=True ] b d e
          in (c, f, g) }""", lambda x: hcb.id_print(x + x), [0])

  def test_simple_outfeed_without_input_token(self):
    self.assertRewrite("""
        { lambda  ; a b.
          let e = create_token a b
              f = create_token a b
              c = add a b
              d g h = outside_call[ arg_treedef=*
                                    callback=...
                                    has_token=True
                                    identity=True ] c e f
          in (d,) }""", lambda x1, x2: hcb.id_print(x1 + x2), [1, 2],
                       has_input_token=False, has_output_token=False)

  def test_simple_outfeed_without_input_token_nor_invars(self):
    self.assertRewrite("""
        { lambda  ; .
          let b = create_token
              c = create_token
              a d e = outside_call[ arg_treedef=*
                                    callback=...
                                    has_token=True
                                    identity=True ] 42 b c
          in (a,) }""", lambda: hcb.id_print(42), [],
                       has_input_token=False, has_output_token=False)

  def test_multiple_tap_without_dependencies(self):
    def f(x):
      hcb.id_print(x, what="x")
      hcb.id_print(x + 1, what="x + 1")
      return 2

    self.assertRewrite("""
        { lambda  ; a c d.
          let _ e f = outside_call[ arg_treedef=*
                                    callback=...
                                    has_token=True
                                    identity=True ] a c d
              b = add a 1
              _ g h = outside_call[ arg_treedef=*
                                    callback=...
                                    has_token=True
                                    identity=True ] b e f
          in (2, g, h) }""", f, [1])

  def test_cond(self):
    y = jnp.ones(5)  # captured const

    def func(x, z):
      return lax.cond(z > 0, (1, 2), lambda a: (a[0], jnp.zeros(5)),
                      z, lambda a: (hcb.id_print(a), y))

    self.assertRewrite("""
        { lambda a ; b c h i.
          let d = gt c 0
              e = convert_element_type[ new_dtype=int32 ] d
              f g j k =
                cond[ branches=( { lambda  ; a b c d f g.
                                   let e h i = outside_call[ arg_treedef=*
                                                             callback=...
                                                             has_token=True
                                                             identity=True ] d f g
                                   in (e, a, h, i) }
                                 { lambda  ; f_ a b c g h.
                                   let d = broadcast_in_dim[ broadcast_dimensions=(  )
                                                             shape=(5,) ] 0.00
                                   in (a, d, g, h) } )
                      linear=(False, False, False, False, False, False) ] e a 1 2 c h i
          in (f, g, j, k) }""", func, [y, 5])

  def test_while(self):
    ct_body = jnp.ones(5, np.float32)  # captured const for the body
    ct_cond = jnp.ones(5, np.float32)  # captured const for the conditional

    def func(x):
      # x: f32[5]
      # c: (f32[5], f32)
      return lax.while_loop(lambda c: c[1] < jnp.sum(c[0] + ct_cond),
                            lambda c: (ct_body, hcb.id_print(c[1]) + 1.),
                            (x, np.float32(1.)))

    self.assertRewrite("""
        { lambda a b ; c f g.
          let d e h i =
                while[ body_jaxpr={ lambda  ; a b c f g.
                                    let d h i = outside_call[ arg_treedef=*
                                                              callback=...
                                                              has_token=True
                                                              identity=True ] c f g
                                        e = add d 1.00
                                    in (a, e, h, i) }
                       body_nconsts=1
                       cond_jaxpr={ lambda  ; a b c g h.
                                    let d = add b a
                                        e = reduce_sum[ axes=(0,) ] d
                                        f = lt c e
                                    in (f,) }
                       cond_nconsts=1 ] a b c 1.00 f g
          in (d, e, h, i) }""", func, [ct_body])

  def test_while_pred_outfeed(self):
    """A while with outfeed in the pred."""
    ct_body = jnp.ones(5)  # captured const for the body
    ct_cond = jnp.ones(2)  # captured const for the conditional

    def func(x):
      return lax.while_loop(lambda c: hcb.id_print(ct_cond, result=c[1]) < 5,
                            lambda c: (ct_body, hcb.id_print(c[1]) + 1),
                            (x, 1))

    self.assertRewrite("""
        { lambda a b ; c f g.
          let j k l = xla_call[ call_jaxpr={ lambda  ; a b c g h.
                                             let d i j = outside_call[ arg_treedef=*
                                                                       callback=...
                                                                       has_token=True
                                                                       identity=True ] a g h
                                                 e = id_tap_dep c d
                                                 f = lt e 5
                                             in (f, i, j) }
                                donated_invars=(False, False, False, False, False)
                                name=cond_before ] a c 1 f g
              bf d e h i =
                while[ body_jaxpr={ lambda  ; r s t u v w x.
                                    let y z ba bb =
                                          xla_call[ call_jaxpr={ lambda  ; a b c f g.
                                                                 let d h i = outside_call[ arg_treedef=*
                                                                                           callback=...
                                                                                           has_token=True
                                                                                           identity=True ] c f g
                                                                     e = add d 1
                                                                 in (a, e, h, i) }
                                                    donated_invars=(False, False, False, False, False)
                                                    name=body ] s u v w x
                                        bc bd be =
                                          xla_call[ call_jaxpr={ lambda  ; a b c g h.
                                                                 let d i j = outside_call[ arg_treedef=*
                                                                                           callback=...
                                                                                           has_token=True
                                                                                           identity=True ] a g h
                                                                     e = id_tap_dep c d
                                                                     f = lt e 5
                                                                 in (f, i, j) }
                                                    donated_invars=(False, False, False, False, False)
                                                    name=cond_body ] r y z ba bb
                                    in (bc, y, z, bd, be) }
                       body_nconsts=2
                       cond_jaxpr={ lambda  ; m n o p q.
                                    let
                                    in (m,) }
                       cond_nconsts=0 ] a b j c 1 k l
          in (d, e, h, i) }""", func, [ct_body])

  def test_scan(self):
    y = jnp.ones(5)  # captured const

    def func(x):
      return lax.scan(lambda c, a: (hcb.id_print(c), y), (1, 2), x)

    self.assertRewrite("""
        { lambda a ; b f g.
          let c d h i e =
                scan[ jaxpr={ lambda  ; a b c g h d.
                              let e f i j =
                                    outside_call[ arg_treedef=PyTreeDef(tuple, [*,*])
                                                  callback=...
                                                  has_token=True
                                                  identity=True ] b c g h
                              in (e, f, i, j, a) }
                      length=5
                      linear=(False, False, False, False, False, False)
                      num_carry=4
                      num_consts=1
                      reverse=False
                      unroll=1 ] a 1 2 f g b
          in (c, d, e, h, i) }""", func, [y])

  def test_scan_custom_jvp(self):
    """custom JVP, inside scan.
    This exercises the custom_jvp_call_jaxpr primitives."""

    @jax.custom_jvp
    def f(x):
      return x * hcb.id_print(x)

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * hcb.id_print(x_dot)
      return primal_out, tangent_out

    def g(x):
      # Sum f(x_i)
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((5,), 0.7)
    self.assertRewrite("""
        { lambda  ; a c d.
          let b e f _ =
                scan[ jaxpr={ lambda  ; a e f b.
                              let c g h = custom_jvp_call_jaxpr[ fun_jaxpr={ lambda  ; a d e.
                                                                             let b f g = outside_call[ arg_treedef=*
                                                                                                       callback=...
                                                                                                       has_token=True
                                                                                                       identity=True ] a d e
                                                                                 c = mul a b
                                                                             in (c, f, g) }
                                                                 num_consts=0 ] b e f
                                  d = add a c
                              in (d, g, h, 0.00) }
                      length=5
                      linear=(False, False, False, False)
                      num_carry=3
                      num_consts=0
                      reverse=False
                      unroll=1 ] 0.00 c d a
          in (b, e, f) }""", g, [arg])
    self.assertRewrite("""
        { lambda  ; a d e.
          let _ _ f g _ b =
                scan[ jaxpr={ lambda  ; a b h i c d.
                              let e j k = custom_jvp_call_jaxpr[ fun_jaxpr={ lambda  ; a d e.
                                                                             let b f g = outside_call[ arg_treedef=*
                                                                                                       callback=...
                                                                                                       has_token=True
                                                                                                       identity=True ] a d e
                                                                                 c = mul a b
                                                                             in (c, f, g) }
                                                                 num_consts=0 ] c h i
                                  f = add a e
                                  g = mul c 3.00
                              in (f, *, j, k, 0.00, g) }
                      length=5
                      linear=(False, True, False, False, False, True)
                      num_carry=4
                      num_consts=0
                      reverse=False
                      unroll=1 ] 0.00 * d e a *
              _ _ h i _ c =
                scan[ jaxpr={ lambda  ; a b g h c d.
                              let e = mul b d
                                  f i j = outside_call[ arg_treedef=*
                                                        callback=...
                                                        has_token=True
                                                        identity=True
                                                        transforms=(('transpose',),) ] e g h
                              in (*, b, i, j, *, f) }
                      length=5
                      linear=(True, True, False, False, True, False)
                      num_carry=4
                      num_consts=0
                      reverse=True
                      unroll=1 ] * 1.00 f g * b
          in (c, h, i) }""", jax.grad(g), [arg])

  def test_scan_custom_vjp(self):
    """custom VJP, inside scan.
    This exercises the custom_vjp_call_jaxpr primitives."""

    @jax.custom_vjp
    def f(x):
      return x * hcb.id_print(x)

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x

    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * hcb.id_print(ct_b),

    f.defvjp(f_fwd, f_bwd)

    def g(x):
      # Sum f(x_i)
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((2,), 0.7)
    self.assertRewrite("""
        { lambda  ; a c d.
          let b e f _ =
                scan[ jaxpr={ lambda  ; a e f b.
                              let c g h = custom_vjp_call_jaxpr[
                                                                 fun_jaxpr={ lambda  ; a d e.
                                                                             let b f g = outside_call[ arg_treedef=*
                                                                                                       callback=...
                                                                                                       has_token=True
                                                                                                       identity=True ] a d e
                                                                                 c = mul a b
                                                                             in (c, f, g) }
                                                                 num_consts=0
                                                                 ] b e f
                                  d = add a c
                              in (d, g, h, 0.00) }
                      length=2
                      linear=(False, False, False, False)
                      num_carry=3
                      num_consts=0
                      reverse=False
                      unroll=1 ] 0.00 c d a
          in (b, e, f) }""", g, [arg])
    self.assertRewrite("""
        { lambda  ; a d e.
          let _ _ f g _ b =
                scan[ jaxpr={ lambda  ; a b h i c d.
                              let e j k = custom_vjp_call_jaxpr[
                                                                 fun_jaxpr={ lambda  ; a d e.
                                                                             let b f g = outside_call[ arg_treedef=*
                                                                                                       callback=...
                                                                                                       has_token=True
                                                                                                       identity=True ] a d e
                                                                                 c = mul a b
                                                                             in (c, f, g) }
                                                                 num_consts=0
                                                                 ] c h i
                                  f = add a e
                                  g = mul c 3.00
                              in (f, *, j, k, 0.00, g) }
                      length=2
                      linear=(False, True, False, False, False, True)
                      num_carry=4
                      num_consts=0
                      reverse=False
                      unroll=1 ] 0.00 * d e a *
              _ _ h i _ c =
                scan[ jaxpr={ lambda  ; a b g h c d.
                              let e i j = outside_call[ arg_treedef=*
                                                        callback=...
                                                        has_token=True
                                                        identity=True ] b g h
                                  f = mul d e
                              in (*, b, i, j, *, f) }
                      length=2
                      linear=(True, True, False, False, True, False)
                      num_carry=4
                      num_consts=0
                      reverse=True
                      unroll=1 ] * 1.00 f g * b
          in (c, h, i) }""", jax.grad(g), [arg])

  def test_remat_loop(self):
    def f(k, x):
      x = hcb.id_print(k + x)
      return -k * x

    def loss(k):
      return lax.fori_loop(0, 1, jax.remat(f), k)

    self.assertRewrite("""
        { lambda  ; a c d.
          let _ _ b e f =
                while[ body_jaxpr={ lambda  ; a b c f g.
                                    let d = add a 1
                                        e h i = remat_call[ call_jaxpr={ lambda  ; a b g h.
                                                                         let c = add a b
                                                                             d i j = outside_call[ arg_treedef=*
                                                                                                   callback=...
                                                                                                   has_token=True
                                                                                                   identity=True ] c g h
                                                                             e = neg a
                                                                             f = mul e d
                                                                         in (f, i, j) }
                                                            concrete=False
                                                            name=f ] a c f g
                                    in (d, b, e, h, i) }
                       body_nconsts=0
                       cond_jaxpr={ lambda  ; a b c e f.
                                    let d = lt a b
                                    in (d,) }
                       cond_nconsts=0 ] 0 1 a c d
          in (b, e, f) }""", loss, [2])

  def test_named_call(self):
    def tap_scalar(init, do_print=False):
      @partial(jax.named_call, name="step")
      def step(acc, step_nr):
        acc = acc + step_nr
        maybe_print(do_print, step_nr, what="step_nr")
        return acc, None

      return lax.scan(step, init, np.arange(2, dtype=np.int32))

    self.assertRewrite("""
        { lambda a ; b d e.
          let c = scan[ jaxpr={ lambda  ; a b.
                                let c = named_call[ call_jaxpr={ lambda  ; a b.
                                                                 let c = add a b
                                                                 in (c,) }
                                                    name=step ] a b
                                in (c,) }
                        length=2
                        linear=(False, False)
                        num_carry=1
                        num_consts=0
                        reverse=False
                        unroll=1 ] b a
          in (c, d, e) }""", tap_scalar, [np.int32(3)])

  def test_pmap(self):
    def f(xv):
      jax.pmap(lambda x: jnp.sin(hcb.id_print(x, tap_with_device=True)),
               axis_name="i")(xv)

    self.assertRewrite("""
        { lambda  ; a b c.
          let _ d e = xla_pmap[ axis_name=i
                                axis_size=1
                                backend=None
                                call_jaxpr={ lambda  ; a d e.
                                             let b f g = outside_call[ arg_treedef=*
                                                                       callback=...
                                                                       has_token=True
                                                                       identity=True ] a d e
                                                 c = sin b
                                             in (c, f, g) }
                                devices=None
                                donated_invars=(False, False, False)
                                global_arg_shapes=(None,)
                                global_axis_size=None
                                in_axes=(0, 0, 0)
                                name=<lambda>
                                out_axes=(0, 0, 0) ] a b c
          in (d, e) }""", f, [np.array([2.], dtype=np.float32)])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
