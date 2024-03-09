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

import contextlib
import io
import logging
import os
import platform
import subprocess
import sys
import textwrap
import unittest

import jax
from jax import config
import jax._src.test_util as jtu
from jax._src import xla_bridge

# Note: importing absltest causes an extra absl root log handler to be
# registered, which causes extra debug log messages. We don't expect users to
# import absl logging, so it should only affect this test. We need to use
# absltest.main and config.parse_flags_with_absl() in order for jax_test flag
# parsing to work correctly with bazel (otherwise we could avoid importing
# absltest/absl logging altogether).
from absl.testing import absltest
config.parse_flags_with_absl()


@contextlib.contextmanager
def capture_jax_logs():
  log_output = io.StringIO()
  handler = logging.StreamHandler(log_output)
  logger = logging.getLogger("jax")

  logger.addHandler(handler)
  try:
    yield log_output
  finally:
    logger.removeHandler(handler)


class LoggingTest(jtu.JaxTestCase):

  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_no_log_spam(self):
    if jtu.is_cloud_tpu() and xla_bridge._backends:
      raise self.skipTest(
          "test requires fresh process on Cloud TPU because only one process "
          "can use the TPU at a time")
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    program = textwrap.dedent("""
      import jax
      jax.device_count()
      f = jax.jit(lambda x: x + 1)
      f(1)
      f(2)
      jax.numpy.add(1, 1)
    """)
    python = sys.executable
    assert "python" in python
    env = os.environ.copy()
    env['TF_CPP_MIN_LOG_LEVEL'] = "1"
    # Make sure C++ logging is at default level for the test process.
    proc = subprocess.run([python, "-c", program], capture_output=True,
                          env=env)

    lines = proc.stdout.split(b"\n")
    lines.extend(proc.stderr.split(b"\n"))
    allowlist = [
        b"",
        b"An NVIDIA GPU may be present on this machine, but a CUDA-enabled "
        b"jaxlib is not installed. Falling back to cpu.",
    ]
    lines = [l for l in lines if l not in allowlist]
    self.assertEmpty(lines)

  def test_debug_logging(self):
    # Warmup so we don't get "No GPU/TPU" warning later.
    jax.jit(lambda x: x + 1)(1)

    # Nothing logged by default (except warning messages, which we don't expect
    # here).
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertEmpty(log_output.getvalue())

    # Turn on all debug logging.
    config.update("jax_debug_log_modules", "jax")
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertIn("Finished tracing + transforming", log_output.getvalue())
    self.assertIn("Compiling <lambda>", log_output.getvalue())

    # Turn off all debug logging.
    config.update("jax_debug_log_modules", None)
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertEmpty(log_output.getvalue())

    # Turn on one module.
    config.update("jax_debug_log_modules", "jax._src.dispatch")
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertIn("Finished tracing + transforming", log_output.getvalue())
    self.assertNotIn("Compiling <lambda>", log_output.getvalue())

    # Turn everything off again.
    config.update("jax_debug_log_modules", None)
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertEmpty(log_output.getvalue())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
