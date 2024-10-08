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
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest

import jax
import jax._src.test_util as jtu
from jax._src import xla_bridge

# Note: importing absltest causes an extra absl root log handler to be
# registered, which causes extra debug log messages. We don't expect users to
# import absl logging, so it should only affect this test. We need to use
# absltest.main and config.parse_flags_with_absl() in order for jax_test flag
# parsing to work correctly with bazel (otherwise we could avoid importing
# absltest/absl logging altogether).
from absl.testing import absltest
jax.config.parse_flags_with_absl()


@contextlib.contextmanager
def jax_debug_log_modules(value):
  # jax_debug_log_modules doesn't have a context manager, because it's
  # not thread-safe. But since tests are always single-threaded, we
  # can define one here.
  original_value = jax.config.jax_debug_log_modules
  jax.config.update("jax_debug_log_modules", value)
  try:
    yield
  finally:
    jax.config.update("jax_debug_log_modules", original_value)

@contextlib.contextmanager
def jax_logging_level(value):
  # jax_logging_level doesn't have a context manager, because it's
  # not thread-safe. But since tests are always single-threaded, we
  # can define one here.
  original_value = jax.config.jax_logging_level
  jax.config.update("jax_logging_level", value)
  try:
    yield
  finally:
    jax.config.update("jax_logging_level", original_value)


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

def _get_repeated_log_fraction(logs: list[str]):
  repeats = 0
  for i in range(len(logs) - 1):
    if logs[i] in logs[i+1] or logs[i+1] in logs[i]:
        repeats += 1
  return repeats / max(len(logs) - 1, 1)


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

    # Save script in file to fix the problem with
    # `tsl::Env::Default()->GetExecutablePath()` not working properly with
    # command flag.
    with tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8", suffix=".py"
    ) as f:
      f.write(textwrap.dedent("""
        import jax
        jax.device_count()
        f = jax.jit(lambda x: x + 1)
        f(1)
        f(2)
        jax.numpy.add(1, 1)
    """))
      python = sys.executable
      assert "python" in python
      env_variables = {}
      if os.getenv("ASAN_OPTIONS"):
        env_variables["ASAN_OPTIONS"] = os.getenv("ASAN_OPTIONS")
      if os.getenv("PYTHONPATH"):
        env_variables["PYTHONPATH"] = os.getenv("PYTHONPATH")
      if os.getenv("LD_LIBRARY_PATH"):
        env_variables["LD_LIBRARY_PATH"] = os.getenv("LD_LIBRARY_PATH")
      if os.getenv("LD_PRELOAD"):
        env_variables["LD_PRELOAD"] = os.getenv("LD_PRELOAD")
      # Make sure C++ logging is at default level for the test process.
      proc = subprocess.run(
          [python, f.name],
          capture_output=True,
          env=env_variables,
      )

      lines = proc.stdout.split(b"\n")
      lines.extend(proc.stderr.split(b"\n"))
      allowlist = [
          b"",
          (
              b"An NVIDIA GPU may be present on this machine, but a"
              b" CUDA-enabled jaxlib is not installed. Falling back to cpu."
          ),
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
    with jax_debug_log_modules("jax"):
      with capture_jax_logs() as log_output:
        jax.jit(lambda x: x + 1)(1)
      self.assertIn("Finished tracing + transforming", log_output.getvalue())
      self.assertIn("Compiling <lambda>", log_output.getvalue())

    # Turn off all debug logging.
    with jax_debug_log_modules(""):
      with capture_jax_logs() as log_output:
        jax.jit(lambda x: x + 1)(1)
      self.assertEmpty(log_output.getvalue())

    # Turn on one module.
    with jax_debug_log_modules("jax._src.dispatch"):
      with capture_jax_logs() as log_output:
        jax.jit(lambda x: x + 1)(1)
      self.assertIn("Finished tracing + transforming", log_output.getvalue())
      self.assertNotIn("Compiling <lambda>", log_output.getvalue())

    # Turn everything off again.
    with jax_debug_log_modules(""):
      with capture_jax_logs() as log_output:
        jax.jit(lambda x: x + 1)(1)
      self.assertEmpty(log_output.getvalue())

  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_stderr_logging(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    program = """
    import jax  # this prints INFO logging from backend imports
    jax.jit(lambda x: x)(1)  # this prints logs to DEBUG (from compilation)
    """

    # strip the leading whitespace from the program script
    program = re.sub(r"^\s+", "", program, flags=re.MULTILINE)

    # test INFO
    cmd = shlex.split(f"env JAX_LOGGING_LEVEL=INFO {sys.executable} -c"
                      f" '{program}'")
    p = subprocess.run(cmd, capture_output=True, text=True)
    log_output = p.stderr
    info_lines = log_output.split("\n")
    self.assertGreater(len(info_lines), 0)
    self.assertIn("INFO", log_output)
    self.assertNotIn("DEBUG", log_output)

    # test DEBUG
    cmd = shlex.split(f"env JAX_LOGGING_LEVEL=DEBUG {sys.executable} -c"
                      f" '{program}'")
    p = subprocess.run(cmd, capture_output=True, text=True)
    log_output = p.stderr
    debug_num_lines = log_output.split("\n")
    self.assertGreater(len(info_lines), 0)
    self.assertIn("INFO", log_output)
    self.assertIn("DEBUG", log_output)
    self.assertGreater(len(debug_num_lines), len(info_lines))

  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_toggling_logging_level(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    _separator = "---------------------------"
    program = f"""
    import sys
    import jax  # this prints INFO logging from backend imports
    jax.jit(lambda x: x)(1)  # this prints logs to DEBUG (from compilation)
    jax.config.update("jax_logging_level", None)
    sys.stderr.write("{_separator}")
    jax.jit(lambda x: x)(1)  # this prints logs to DEBUG (from compilation)
    """

    # strip the leading whitespace from the program script
    program = re.sub(r"^\s+", "", program, flags=re.MULTILINE)

    cmd = shlex.split(f"env JAX_LOGGING_LEVEL=DEBUG {sys.executable} -c"
                      f" '{program}'")
    p = subprocess.run(cmd, capture_output=True, text=True)
    log_output = p.stderr
    m = re.search(_separator, log_output)
    self.assertTrue(m is not None)
    log_output_verbose = log_output[:m.start()]
    log_output_silent = log_output[m.end():]

    self.assertIn("Finished tracing + transforming <lambda> for pjit",
                  log_output_verbose)
    self.assertEqual(log_output_silent, "")

  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_double_logging_absent(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    program = """
    import jax  # this prints INFO logging from backend imports
    jax.jit(lambda x: x)(1)  # this prints logs to DEBUG (from compilation)
    """

    # strip the leading whitespace from the program script
    program = re.sub(r"^\s+", "", program, flags=re.MULTILINE)

    cmd = shlex.split(f"env JAX_LOGGING_LEVEL=DEBUG {sys.executable} -c"
                      f" '{program}'")
    p = subprocess.run(cmd, capture_output=True, text=True)
    log_output = p.stderr
    self.assertNotEmpty(log_output)
    log_lines = log_output.strip().split("\n")
    self.assertLess(_get_repeated_log_fraction(log_lines), 0.2)

  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_cpp_logging_level(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    program = f"""
    import sys
    import jax  # this prints INFO logging from backend imports
    jax.distributed.initialize("127.0.0.1:12345", num_processes=1, process_id=0)
    """

    # strip the leading whitespace from the program script
    program = re.sub(r"^\s+", "", program, flags=re.MULTILINE)

    # verbose logging: DEBUG, VERBOSE
    cmd = shlex.split(f"env JAX_LOGGING_LEVEL=DEBUG {sys.executable} -c"
                      f" '{program}'")
    p = subprocess.run(cmd, capture_output=True, text=True)
    self.assertIn("Initializing CoordinationService", p.stderr)

    cmd = shlex.split(f"env JAX_LOGGING_LEVEL=INFO {sys.executable} -c"
                      f" '{program}'")
    p = subprocess.run(cmd, capture_output=True, text=True)
    self.assertIn("Initializing CoordinationService", p.stderr)

    # verbose logging: WARNING, None
    cmd = shlex.split(f"env JAX_LOGGING_LEVEL=WARNING {sys.executable} -c"
                      f" '{program}'")
    p = subprocess.run(cmd, capture_output=True, text=True)
    self.assertNotIn("Initializing CoordinationService", p.stderr)

    cmd = shlex.split(f"{sys.executable} -c"
                      f" '{program}'")
    p = subprocess.run(cmd, capture_output=True, text=True)
    self.assertNotIn("Initializing CoordinationService", p.stderr)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
