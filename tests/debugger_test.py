# Copyright 2022 The JAX Authors.
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

from collections.abc import Sequence
import contextlib
import io
import re
import textwrap
from typing import IO
import unittest

from absl.testing import absltest
import jax
from jax.experimental import pjit
from jax._src import debugger
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

def make_fake_stdin_stdout(commands: Sequence[str]) -> tuple[IO[str], io.StringIO]:
  fake_stdin = io.StringIO()
  fake_stdin.truncate(0)
  for command in commands:
    fake_stdin.write(command + "\n")
  fake_stdin.seek(0)
  return fake_stdin, io.StringIO()

def _format_multiline(text):
  return textwrap.dedent(text).lstrip()

_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(2))

def tearDownModule():
  _exit_stack.close()

foo = 2

class CliDebuggerTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu", "tpu"]):
      self.skipTest(f"Host callback not supported on {jtu.device_under_test()}")

  def test_debugger_eof(self):
    stdin, stdout = make_fake_stdin_stdout([])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y
    with self.assertRaises(SystemExit):
      f(2.)
      jax.effects_barrier()

  def test_debugger_can_continue(self):
    stdin, stdout = make_fake_stdin_stdout(["c"])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y
    f(2.)
    jax.effects_barrier()
    expected = _format_multiline(r"""
    Entering jdb:
    (jdb) """)
    self.assertEqual(stdout.getvalue(), expected)

  def test_debugger_can_print_value(self):
    stdin, stdout = make_fake_stdin_stdout(["p x", "c"])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y
    expected = _format_multiline(r"""
    Entering jdb:
    (jdb) Array(2., dtype=float32)
    (jdb) """)
    f(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  def test_debugger_can_print_value_in_jit(self):
    stdin, stdout = make_fake_stdin_stdout(["p x", "c"])

    @jax.jit
    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y
    expected = _format_multiline(r"""
    Entering jdb:
    (jdb) Array(2., dtype=float32)
    (jdb) """)
    f(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  def test_debugger_can_print_multiple_values(self):
    stdin, stdout = make_fake_stdin_stdout(["p x, y", "c"])

    @jax.jit
    def f(x):
      y = x + 1.
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y
    expected = _format_multiline(r"""
    Entering jdb:
    (jdb) (Array(2., dtype=float32), Array(3., dtype=float32))
    (jdb) """)
    f(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  def test_debugger_can_print_context(self):
    stdin, stdout = make_fake_stdin_stdout(["l", "c"])

    @jax.jit
    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y
    f(2.)
    jax.effects_barrier()
    expected = _format_multiline(r"""
    Entering jdb:
    \(jdb\) > .*debugger_test\.py\([0-9]+\)
            @jax\.jit
            def f\(x\):
              y = jnp\.sin\(x\)
    ->        debugger\.breakpoint\(stdin=stdin, stdout=stdout, backend="cli"\)
              return y
    .*
    \(jdb\) """)
    self.assertRegex(stdout.getvalue(), expected)

  def test_debugger_can_print_backtrace(self):
    stdin, stdout = make_fake_stdin_stdout(["bt", "c"])

    @jax.jit
    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y
    expected = _format_multiline(r"""
    Entering jdb:.*
    \(jdb\) Traceback:.*
    """)
    f(2.)
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

  def test_debugger_can_work_with_multiple_stack_frames(self):
    stdin, stdout = make_fake_stdin_stdout(["l", "u", "p x", "d", "c"])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y

    @jax.jit
    def g(x):
      y = f(x)
      return jnp.exp(y)
    expected = _format_multiline(r"""
    Entering jdb:
    \(jdb\) > .*debugger_test\.py\([0-9]+\)
            def f\(x\):
              y = jnp\.sin\(x\)
    ->        debugger\.breakpoint\(stdin=stdin, stdout=stdout, backend="cli"\)
              return y
    .*
    \(jdb\) > .*debugger_test\.py\([0-9]+\).*
            @jax\.jit
            def g\(x\):
    ->        y = f\(x\)
              return jnp\.exp\(y\)
    .*
    \(jdb\) Array\(2\., dtype=float32\)
    \(jdb\) > .*debugger_test\.py\([0-9]+\)
            def f\(x\):
              y = jnp\.sin\(x\)
    ->        debugger\.breakpoint\(stdin=stdin, stdout=stdout, backend="cli"\)
              return y
    .*
    \(jdb\) """)
    g(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

  def test_can_use_multiple_breakpoints(self):
    stdin, stdout = make_fake_stdin_stdout(["p y", "c", "p y", "c"])

    def f(x):
      y = x + 1.
      debugger.breakpoint(stdin=stdin, stdout=stdout, ordered=True,
          backend="cli")
      return y

    @jax.jit
    def g(x):
      y = f(x) * 2.
      debugger.breakpoint(stdin=stdin, stdout=stdout, ordered=True,
          backend="cli")
      return jnp.exp(y)
    expected = _format_multiline(r"""
    Entering jdb:
    (jdb) Array(3., dtype=float32)
    (jdb) Entering jdb:
    (jdb) Array(6., dtype=float32)
    (jdb) """)
    g(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  def test_debugger_works_with_vmap(self):
    stdin, stdout = make_fake_stdin_stdout(["p y", "c", "p y", "c"])

    def f(x):
      y = x + 1.
      debugger.breakpoint(stdin=stdin, stdout=stdout, ordered=True,
          backend="cli")
      return 2. * y

    @jax.jit
    @jax.vmap
    def g(x):
      y = f(x)
      return jnp.exp(y)
    expected = _format_multiline(r"""
    Entering jdb:
    (jdb) Array(1., dtype=float32)
    (jdb) Entering jdb:
    (jdb) Array(2., dtype=float32)
    (jdb) """)
    g(jnp.arange(2., dtype=jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  def test_debugger_works_with_pmap(self):
    if jax.local_device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")

    stdin, stdout = make_fake_stdin_stdout(["p y", "c", "p y", "c"])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y

    @jax.pmap
    def g(x):
      y = f(x)
      return jnp.exp(y)
    expected = _format_multiline(r"""
    Entering jdb:
    \(jdb\) Array\(.*, dtype=float32\)
    \(jdb\) Entering jdb:
    \(jdb\) Array\(.*, dtype=float32\)
    \(jdb\) """)
    g(jnp.arange(2., dtype=jnp.float32))
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

  def test_debugger_works_with_pjit(self):
    if jax.default_backend() != "tpu":
      raise unittest.SkipTest("`pjit` doesn't work with CustomCall.")

    stdin, stdout = make_fake_stdin_stdout(["p y", "c"])

    def f(x):
      y = x + 1
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      return y

    def g(x):
      y = f(x)
      return jnp.exp(y)
    g = pjit.pjit(
        g,
        in_shardings=jax.sharding.PartitionSpec("dev"),
        out_shardings=jax.sharding.PartitionSpec("dev"),
    )
    with jax.sharding.Mesh(np.array(jax.devices()), ["dev"]):
      arr = (1 + jnp.arange(8)).astype(np.int32)
      expected = _format_multiline(r"""
      Entering jdb:
      \(jdb\) {}
      \(jdb\) """.format(re.escape(repr(arr))))
      g(jnp.arange(8, dtype=jnp.int32))
      jax.effects_barrier()
      self.assertRegex(stdout.getvalue(), expected)

  def test_debugger_uses_local_before_global_scope(self):
    stdin, stdout = make_fake_stdin_stdout(["p foo", "c"])

    foo = "outer"

    def f(x):
      foo = "inner"
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      del foo
      return x

    del foo
    expected = _format_multiline(r"""
    Entering jdb:
    \(jdb\) 'inner'
    \(jdb\) """)
    f(2.)
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

  def test_debugger_accesses_globals(self):
    stdin, stdout = make_fake_stdin_stdout(["p foo", "c"])

    @jax.jit
    def g():
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")

    expected = _format_multiline(r"""
    Entering jdb:
    \(jdb\) \*\*\* NameError: name 'foo' is not defined
    \(jdb\) """)
    g()
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

  def test_can_limit_num_frames(self):
    stdin, stdout = make_fake_stdin_stdout(["u", "p x", "c"])

    def g():
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli",
                          num_frames=2)

    @jax.jit
    def f():
      x = 2
      g()
      return x
    _ = f()
    expected = _format_multiline(r"""
    Entering jdb:
    \(jdb\) .*
    .*
    .*
    .*
    .*
    .*
    .*
    \(jdb\) 2
    \(jdb\) """)
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

    stdin, stdout = make_fake_stdin_stdout(["u", "u", "c"])

    def g2():
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli",
                          num_frames=2)

    @jax.jit
    def f2():
      x = 2
      g2()
      return x

    expected = ".*At topmost frame.*"
    _ = f2()
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

  def test_can_handle_dictionaries_with_unsortable_keys(self):
    stdin, stdout = make_fake_stdin_stdout(["p x", "p weird_dict",
                                            "p weirder_dict", "c"])

    @jax.jit
    def f():
      weird_dict = {(lambda x: x): 2., (lambda x: x * 2): 3}
      weirder_dict = {(lambda x: x): weird_dict}
      x = 2.
      debugger.breakpoint(stdin=stdin, stdout=stdout, backend="cli")
      del weirder_dict
      return x
    expected = _format_multiline(r"""
    Entering jdb:
    \(jdb\) 2.0
    \(jdb\) <cant_flatten>
    \(jdb\) <cant_flatten>
    \(jdb\) """)
    _ = f()
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
