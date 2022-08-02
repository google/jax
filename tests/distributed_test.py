# Copyright 2022 Google LLC
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

import os
import subprocess
import sys
import threading
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax.config import config
from jax._src import distributed
import jax._src.lib
from jax._src import test_util as jtu

try:
  import portpicker
except ImportError:
  portpicker = None

config.parse_flags_with_absl()


@unittest.skipIf(not portpicker, "Test requires portpicker")
class DistributedTest(jtu.JaxTestCase):

  # TODO(phawkins): Enable after https://github.com/google/jax/issues/11222
  # is fixed.
  @unittest.SkipTest
  def testInitializeAndShutdown(self):
    # Tests the public APIs. Since they use global state, we cannot use
    # concurrency to simulate multiple tasks.
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(coordinator_address=f"localhost:{port}",
                               num_processes=1,
                               process_id=0)
    jax.distributed.shutdown()


  @parameterized.parameters([1, 2, 4])
  def testConcurrentInitializeAndShutdown(self, n):
    port = portpicker.pick_unused_port()
    def task(i):
      # We can't call the public APIs directly because they use global state.
      state = distributed.State()
      state.initialize(coordinator_address=f"localhost:{port}",
                       num_processes=n,
                       process_id=i)
      state.shutdown()

    threads = [threading.Thread(target=task, args=(i,)) for i in range(n)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()


@unittest.skipIf(not portpicker, "Test requires portpicker")
class MultiProcessGpuTest(jtu.JaxTestCase):

  def test_gpu_distributed_initialize(self):
    if jax.devices()[0].platform != 'gpu':
      raise unittest.SkipTest('Tests only for GPU.')

    port = portpicker.pick_unused_port()
    num_gpus = 4
    num_gpus_per_task = 1
    num_tasks = num_gpus // num_gpus_per_task

    os.environ["JAX_PORT"] = str(port)
    os.environ["NUM_TASKS"] = str(num_tasks)

    subprocesses = []
    for task in range(num_tasks):
      env = os.environ.copy()
      env["TASK"] = str(task)
      env["CUDA_VISIBLE_DEVICES"] = ",".join(
          str((task * num_gpus_per_task) + i) for i in range(num_gpus_per_task))
      args = [
          sys.executable,
          "-c",
          ('import jax, os; '
           'jax.distributed.initialize('
               'f\'localhost:{os.environ["JAX_PORT"]}\', '
               'int(os.environ["NUM_TASKS"]), int(os.environ["TASK"])); '
           'print(f\'{jax.local_device_count()},{jax.device_count()}\', end="")'
          )
      ]
      subprocesses.append(subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True))

    for proc in subprocesses:
      out, err = proc.communicate()
      print(err)
      self.assertEqual(proc.returncode, 0)
      self.assertEqual(out, f'{num_gpus_per_task},{num_gpus}')


@unittest.skipIf(not portpicker, "Test requires portpicker")
class MultiProcessGpuWithEnvTest(jtu.JaxTestCase):

  def test_gpu_distributed_initialize(self):
    if jax.devices()[0].platform != 'gpu':
      raise unittest.SkipTest('Tests only for GPU.')

    port = portpicker.pick_unused_port()
    num_gpus = 4
    num_gpus_per_task = 1
    num_tasks = num_gpus // num_gpus_per_task

    os.environ["JAX_COORDINATOR_ADDRESS"] = f'localhost:{port}'
    os.environ["JAX_NUM_PROCESSES"] = str(num_tasks)

    subprocesses = []
    for task in range(num_tasks):
      env = os.environ.copy()
      env["JAX_PROCESS_ID"] = str(task)
      env["CUDA_VISIBLE_DEVICES"] = ",".join(
          str((task * num_gpus_per_task) + i) for i in range(num_gpus_per_task))
      args = [
          sys.executable,
          "-c",
          ('import jax, os; '
           'jax.distributed.initialize(); '
           'print(f\'{jax.local_device_count()},{jax.device_count()}\', end="")'
          )
      ]
      subprocesses.append(subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True))

    for proc in subprocesses:
      out, err = proc.communicate()
      print(err)
      self.assertEqual(proc.returncode, 0)
      self.assertEqual(out, f'{num_gpus_per_task},{num_gpus}')


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
