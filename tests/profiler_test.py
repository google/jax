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

from functools import partial
import glob
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax.profiler
import jax._src.test_util as jtu
from jax._src import profiler

try:
  import portpicker
except ImportError:
  portpicker = None

try:
  from tensorflow.python.profiler import profiler_client
  from tensorflow.python.profiler import profiler_v2 as tf_profiler
except ImportError:
  profiler_client = None
  tf_profiler = None

TBP_ENABLED = False
try:
  import tensorboard_plugin_profile
  del tensorboard_plugin_profile
  TBP_ENABLED = True
except ImportError:
  pass

jax.config.parse_flags_with_absl()


class ProfilerTest(unittest.TestCase):
  # These tests simply test that the profiler API does not crash; they do not
  # check functional correctness.

  def setUp(self):
    super().setUp()
    self.worker_start = threading.Event()
    self.profile_done = False

  @unittest.skipIf(not portpicker, "Test requires portpicker")
  def testStartStopServer(self):
    port = portpicker.pick_unused_port()
    jax.profiler.start_server(port=port)
    del port
    jax.profiler.stop_server()

  @unittest.skipIf(not portpicker, "Test requires portpicker")
  def testCantStartMultipleServers(self):
    port = portpicker.pick_unused_port()
    jax.profiler.start_server(port=port)
    port = portpicker.pick_unused_port()
    with self.assertRaisesRegex(
        ValueError, "Only one profiler server can be active at a time."):
      jax.profiler.start_server(port=port)
    jax.profiler.stop_server()

  def testCantStopServerBeforeStartingServer(self):
    with self.assertRaisesRegex(ValueError, "No active profiler server."):
      jax.profiler.stop_server()

  def testProgrammaticProfiling(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      try:
        jax.profiler.start_trace(tmpdir)
        jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
            jnp.ones(jax.local_device_count()))
      finally:
        jax.profiler.stop_trace()

      proto_path = glob.glob(os.path.join(tmpdir, "**/*.xplane.pb"),
                             recursive=True)
      self.assertEqual(len(proto_path), 1)
      with open(proto_path[0], "rb") as f:
        proto = f.read()
      # Sanity check that serialized proto contains host, device, and
      # Python traces without deserializing.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"/device:TPU", proto)
      self.assertIn(b"pxla.py", proto)

  def testProfilerGetFDOProfile(self):
    # Tests stop_and_get_fod_profile could run.
    try:
      jax.profiler.start_trace("test")
      jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
          jnp.ones(jax.local_device_count())
      )
    finally:
      fdo_profile = profiler.stop_and_get_fdo_profile()
    if jtu.test_device_matches(["gpu"]) and jtu.is_device_cuda():
      self.assertIn(b"copy", fdo_profile)

  def testProgrammaticProfilingErrors(self):
    with self.assertRaisesRegex(RuntimeError, "No profile started"):
      jax.profiler.stop_trace()

    try:
      with tempfile.TemporaryDirectory() as tmpdir:
        jax.profiler.start_trace(tmpdir)
        with self.assertRaisesRegex(
          RuntimeError,
          "Profile has already been started. Only one profile may be run at a "
          "time."):
          jax.profiler.start_trace(tmpdir)
    finally:
      jax.profiler.stop_trace()

  def testProgrammaticProfilingContextManager(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with jax.profiler.trace(tmpdir):
        jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
            jnp.ones(jax.local_device_count()))

      proto_path = glob.glob(os.path.join(tmpdir, "**/*.xplane.pb"),
                             recursive=True)
      self.assertEqual(len(proto_path), 1)
      with open(proto_path[0], "rb") as f:
        proto = f.read()
      # Sanity check that serialized proto contains host and device traces
      # without deserializing.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"/device:TPU", proto)

  def testTraceAnnotation(self):
    x = 3
    with jax.profiler.TraceAnnotation("mycontext"):
      x = x + 2

  def testTraceFunction(self):
    @jax.profiler.annotate_function
    def f(x, *, y):
      return x + 2 * y
    self.assertEqual(f(7, y=3), 13)

    @jax.profiler.annotate_function
    def f(x, *, name):
      return x + 2 * len(name)
    self.assertEqual(f(7, name="abc"), 13)

    @partial(jax.profiler.annotate_function, name="aname")
    def g(x):
      return x + 2
    self.assertEqual(g(7), 9)

    @partial(jax.profiler.annotate_function, name="aname", akwarg="hello")
    def h(x):
      return x + 2
    self.assertEqual(h(7), 9)

  def testDeviceMemoryProfile(self):
    x = jnp.ones((20,)) + 7.
    self.assertIsInstance(jax.profiler.device_memory_profile(), bytes)
    del x

  def _check_xspace_pb_exist(self, logdir):
    path = os.path.join(logdir, 'plugins', 'profile', '*', '*.xplane.pb')
    self.assertEqual(1, len(glob.glob(path)),
                     'Expected one path match: ' + path)

  @unittest.skip("Test causes OOMs")
  @unittest.skipIf(not (portpicker and profiler_client and tf_profiler),
    "Test requires tensorflow.profiler and portpicker")
  def testSingleWorkerSamplingMode(self, delay_ms=None):
    def on_worker(port, worker_start):
      jax.profiler.start_server(port)
      worker_start.set()
      x = jnp.ones((1000, 1000))
      while True:
        with jax.profiler.TraceAnnotation("atraceannotation"):
          jnp.dot(x, x.T).block_until_ready()
          if self.profile_done:
            jax.profiler.stop_server()
            break

    def on_profile(port, logdir, worker_start):
      worker_start.wait()
      options = tf_profiler.ProfilerOptions(
          host_tracer_level=2,
          python_tracer_level=2,
          device_tracer_level=1,
          delay_ms=delay_ms,
      )

      # Request for 1000 milliseconds of profile.
      duration_ms = 1000
      profiler_client.trace(f'localhost:{port}', logdir, duration_ms,
                            '', 1000, options)
      self.profile_done = True

    logdir = absltest.get_default_test_tmpdir()
    # Remove any existing log files.
    shutil.rmtree(logdir, ignore_errors=True)
    port = portpicker.pick_unused_port()
    thread_profiler = threading.Thread(
        target=on_profile, args=(port, logdir, self.worker_start))
    thread_worker = threading.Thread(
        target=on_worker, args=(port, self.worker_start))
    thread_worker.start()
    thread_profiler.start()
    thread_profiler.join()
    thread_worker.join(120)
    self._check_xspace_pb_exist(logdir)

  @unittest.skipIf(
      not (portpicker and profiler_client and tf_profiler and TBP_ENABLED),
    "Test requires tensorflow.profiler, portpicker and "
    "tensorboard_profile_plugin")
  def test_remote_profiler(self):
    port = portpicker.pick_unused_port()
    jax.profiler.start_server(port)

    logdir = absltest.get_default_test_tmpdir()
    # Remove any existing log files.
    shutil.rmtree(logdir, ignore_errors=True)
    def on_profile():
      os.system(
          f"{sys.executable} -m jax.collect_profile {port} 500 "
          f"--log_dir {logdir} --no_perfetto_link")

    thread_profiler = threading.Thread(
        target=on_profile, args=())
    thread_profiler.start()
    start_time = time.time()
    y = jnp.zeros((5, 5))
    while time.time() - start_time < 10:
      y = jnp.dot(y, y)
    jax.profiler.stop_server()
    thread_profiler.join()
    self._check_xspace_pb_exist(logdir)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
