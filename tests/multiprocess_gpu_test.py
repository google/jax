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

import contextlib
import os
import shutil
import subprocess
import sys
import threading
import unittest
import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
from jax._src import core
from jax._src import distributed
from jax._src import test_util as jtu
from jax._src import util
from jax.experimental import pjit
import jax.numpy as jnp

# Used to test for mpi4py installation and skip tests if not installed
import importlib.util

try:
  import portpicker
except ImportError:
  portpicker = None

jax.config.parse_flags_with_absl()

@unittest.skipIf(not portpicker, "Test requires portpicker")
class DistributedTest(jtu.JaxTestCase):

  # TODO(phawkins): Enable after https://github.com/google/jax/issues/11222
  # is fixed.
  @unittest.SkipTest
  def testInitializeAndShutdown(self):
    if not jtu.test_device_matches(['gpu']):
      self.skipTest('Test only works with GPUs.')
    # Tests the public APIs. Since they use global state, we cannot use
    # concurrency to simulate multiple tasks.
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(coordinator_address=f"localhost:{port}",
                               num_processes=1,
                               process_id=0)
    jax.distributed.shutdown()


  @parameterized.parameters([1, 2, 4])
  def testConcurrentInitializeAndShutdown(self, n):
    if not jtu.test_device_matches(['gpu']):
      self.skipTest('Test only works with GPUs.')
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
    if not jtu.test_device_matches(['gpu']):
      raise unittest.SkipTest('Tests only for GPU.')

    port = portpicker.pick_unused_port()
    num_gpus = 4
    num_gpus_per_task = 1
    num_tasks = num_gpus // num_gpus_per_task

    with contextlib.ExitStack() as exit_stack:
      subprocesses = []
      for task in range(num_tasks):
        env = os.environ.copy()
        env["JAX_PORT"] = str(port)
        env["NUM_TASKS"] = str(num_tasks)
        env["TASK"] = str(task)
        if jtu.is_device_rocm():
          env["HIP_VISIBLE_DEVICES"] = ",".join(
              str((task * num_gpus_per_task) + i) for i in range(num_gpus_per_task))
        else:
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
        proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        subprocesses.append(exit_stack.enter_context(proc))

      try:
        for proc in subprocesses:
          out, _ = proc.communicate()
          self.assertEqual(proc.returncode, 0)
          self.assertEqual(out, f'{num_gpus_per_task},{num_gpus}')
      finally:
        for proc in subprocesses:
          proc.kill()

  def test_distributed_jax_visible_devices(self):
    """Test jax_visible_devices works in distributed settings."""
    if not jtu.test_device_matches(['gpu']):
      raise unittest.SkipTest('Tests only for GPU.')

    port = portpicker.pick_unused_port()
    num_gpus = 4
    num_gpus_per_task = 1
    num_tasks = num_gpus // num_gpus_per_task

    with contextlib.ExitStack() as exit_stack:
      subprocesses = []
      for task in range(num_tasks):
        env = os.environ.copy()
        env["JAX_PORT"] = str(port)
        env["NUM_TASKS"] = str(num_tasks)
        env["TASK"] = str(task)
        visible_devices = ",".join(
            str((task * num_gpus_per_task) + i) for i in range(num_gpus_per_task))

        if jtu.is_device_rocm():
          program = (
            'import jax, os; '
            f'jax.config.update("jax_rocm_visible_devices", "{visible_devices}"); '
            'jax.distributed.initialize('
            'f\'localhost:{os.environ["JAX_PORT"]}\', '
            'int(os.environ["NUM_TASKS"]), int(os.environ["TASK"])); '
            's = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(jax.numpy.ones(jax.local_device_count())); '
            'print(f\'{jax.local_device_count()},{jax.device_count()},{s}\', end=""); '
          )
        else:
          program = (
            'import jax, os; '
            f'jax.config.update("jax_cuda_visible_devices", "{visible_devices}"); '
            'jax.distributed.initialize('
            'f\'localhost:{os.environ["JAX_PORT"]}\', '
            'int(os.environ["NUM_TASKS"]), int(os.environ["TASK"])); '
            's = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(jax.numpy.ones(jax.local_device_count())); '
            'print(f\'{jax.local_device_count()},{jax.device_count()},{s}\', end=""); '
          )
        args = [sys.executable, "-c", program]
        proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        subprocesses.append(exit_stack.enter_context(proc))

      try:
        for proc in subprocesses:
          out, err = proc.communicate()
          self.assertEqual(proc.returncode, 0, msg=f"Process failed:\n\n{err}")
          self.assertRegex(out, f'{num_gpus_per_task},{num_gpus},\\[{num_gpus}.\\]$')
      finally:
        for proc in subprocesses:
          proc.kill()

  def test_gpu_ompi_distributed_initialize(self):
    if not jtu.test_device_matches(['gpu']):
      raise unittest.SkipTest('Tests only for GPU.')
    if shutil.which('mpirun') is None:
      raise unittest.SkipTest('Tests only for MPI (mpirun not found).')

    num_gpus = 4
    num_gpus_per_task = 1

    with contextlib.ExitStack() as exit_stack:
      args = [
          'mpirun',
          '--oversubscribe',
          '--allow-run-as-root',
          '-n',
          str(num_gpus),
          sys.executable,
          '-c',
          ('import jax, os; '
          'jax.distributed.initialize(); '
          'print(f\'{jax.local_device_count()},{jax.device_count()}\' if jax.process_index() == 0 else \'\', end="")'
          )
      ]
      env = os.environ.copy()
      # In case the job was launched via Slurm,
      # prevent OpenMPI from detecting Slurm environment
      env.pop('SLURM_JOBID', None)
      proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)
      proc = exit_stack.enter_context(proc)

      try:
        out, _ = proc.communicate()
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(out, f'{num_gpus_per_task},{num_gpus}')
      finally:
        proc.kill()

  def test_gpu_mpi4py_distributed_initialize(self):
    if not jtu.test_device_matches(['gpu']):
      raise unittest.SkipTest('Tests only for GPU.')
    if shutil.which('mpirun') is None:
      raise unittest.SkipTest('Tests only for MPI (mpirun not found).')
    if importlib.util.find_spec("mpi4py") is None:
      raise unittest.SkipTest('Test of mpi4py initialize only possible with mpi4py installed.')

    num_gpus = 4
    num_gpus_per_task = 1

    with contextlib.ExitStack() as exit_stack:
      args = [
          'mpirun',
          '--oversubscribe',
          '--allow-run-as-root',
          '-n',
          str(num_gpus),
          sys.executable,
          '-c',
          ('import jax, os; '
          'jax.distributed.initialize(spec_detection_method="mpi4py"); '
          'print(f\'{jax.local_device_count()},{jax.device_count()}\' if jax.process_index() == 0 else \'\', end="")'
          )
      ]
      env = os.environ.copy()
      # In case the job was launched via Slurm,
      # prevent OpenMPI from detecting Slurm environment
      env.pop('SLURM_JOBID', None)
      proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)
      proc = exit_stack.enter_context(proc)

      try:
        out, _ = proc.communicate()
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(out, f'{num_gpus_per_task},{num_gpus}')
      finally:
        proc.kill()


@unittest.skipIf(
    os.environ.get("SLURM_JOB_NUM_NODES", None) != "2",
    "Slurm environment with at least two nodes needed!")
@jtu.pytest_mark_if_available('SlurmMultiNodeGpuTest')
@jtu.with_config(experimental_xmap_spmd_lowering=True)
class SlurmMultiNodeGpuTest(jtu.JaxTestCase):

  def sorted_devices(self):
    devices = sorted(jax.devices(), key=lambda d: (d.id, d.host_id))
    if len(devices) != 16:
      raise unittest.SkipTest(
          "Test assumes that it runs on 16 devices (2 nodes)")
    return devices

  def create_2d_non_contiguous_mesh(self):
    devices = self.sorted_devices()
    device_mesh = np.array([[devices[0], devices[2]],
                            [devices[4], devices[6]],
                            [devices[1], devices[3]],
                            [devices[5], devices[7]],
                            [devices[8], devices[10]],
                            [devices[12], devices[14]],
                            [devices[9], devices[11]],
                            [devices[13], devices[15]]])
    # The mesh looks like this (the integers are process index):
    #   0 2
    #   4 6
    #   1 3
    #   5 7
    #   8 10
    #   12 14
    #   9 11
    #   13 15
    assert [d.id for d in device_mesh.flat
           ] == [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
    return jax.sharding.Mesh(device_mesh, ("x", "y"))

  def test_gpu_multi_node_initialize_and_psum(self):

    # Hookup the ENV vars expected to be set already in the SLURM environment
    coordinator_address = os.environ.get("SLURM_STEP_NODELIST", None)
    if coordinator_address is not None and '[' in coordinator_address:
      coordinator_address = coordinator_address.split('[')[0] + \
                            coordinator_address.split('[')[1].split(',')[0]
    num_tasks = os.environ.get("SLURM_NPROCS", None)
    taskid = os.environ.get("SLURM_PROCID", None)
    localid = os.environ.get("SLURM_LOCALID", None)

    # fixing port since it needs to be the same for all the processes
    port = "54321"

    print(f"coord addr:port : {coordinator_address}:{port}\nTotal tasks: "
          f"{num_tasks}\ntask id: {taskid}\nlocal id: {localid}")

    self.assertEqual(
        coordinator_address is None or num_tasks is None or taskid is None,
        False)

    # os.environ["CUDA_VISIBLE_DEVICES"] = localid #WAR for Bug:12119
    jax.config.update("jax_cuda_visible_devices", localid)

    jax.distributed.initialize(coordinator_address=f'{coordinator_address}:{port}',
                               num_processes=int(num_tasks),
                               process_id=int(taskid))

    print(f"Total devices: {jax.device_count()}, Total tasks: {int(num_tasks)}, "
          f"Devices per task: {jax.local_device_count()}")

    self.assertEqual(jax.device_count(),
                     int(num_tasks) * jax.local_device_count())

    x = jnp.ones(jax.local_device_count())
    y = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(x)
    self.assertEqual(y[0], jax.device_count())
    print(y)

  def test_gpu_multi_node_transparent_initialize_and_psum(self):

    jax.distributed.initialize()

    print(f"Total devices: {jax.device_count()}, "
          f"Devices per task: {jax.local_device_count()}")

    self.assertEqual(jax.device_count(), int(os.environ['SLURM_NTASKS']))
    self.assertEqual(jax.local_device_count(), 1)

    x = jnp.ones(jax.local_device_count())
    y = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(x)
    self.assertEqual(y[0], jax.device_count())
    print(y)

  def test_pjit_gda_multi_input_multi_output(self):
    jax.distributed.initialize()
    global_mesh = jtu.create_global_mesh((8, 2), ("x", "y"))
    global_input_shape = (16, 2)
    global_input_data = np.arange(
        util.prod(global_input_shape)).reshape(global_input_shape)

    def cb(index):
      return global_input_data[index]

    mesh_axes1 = jax.sharding.PartitionSpec("x", "y")
    gda1 = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes1), cb)
    mesh_axes2 = jax.sharding.PartitionSpec("x")
    gda2 = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes2), cb)
    mesh_axes3 = jax.sharding.PartitionSpec(("x", "y"))
    gda3 = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes3), cb)

    with jax.sharding.Mesh(global_mesh.devices, global_mesh.axis_names):

      @functools.partial(
          pjit.pjit,
          out_shardings=(mesh_axes1, None, mesh_axes2))
      def f(x, y, z):
        return x @ x.T, y, z

      out1, out2, out3 = f(gda1, gda2, gda3)

      self.assertEqual(out1.shape, (16, 16))
      self.assertEqual(out1.addressable_shards[0].data.shape, (2, 8))
      expected_matrix_mul = global_input_data @ global_input_data.T
      for s in out1.addressable_shards:
        np.testing.assert_array_equal(np.asarray(s.data),
                                      expected_matrix_mul[s.index])

      self.assertEqual(out2.shape, (16, 2))
      self.assertEqual(out2.addressable_shards[0].data.shape, (16, 2))
      for s in out2.addressable_shards:
        np.testing.assert_array_equal(np.asarray(s.data), global_input_data)

      self.assertEqual(out3.shape, (16, 2))
      self.assertEqual(out3.addressable_shards[0].data.shape, (2, 2))
      for s in out3.addressable_shards:
        np.testing.assert_array_equal(np.asarray(s.data),
                                      global_input_data[s.index])

  def test_pjit_gda_non_contiguous_mesh(self):
    jax.distributed.initialize()
    devices = self.sorted_devices()
    mesh_devices = np.array(devices[0:8:2] + devices[1:8:2] + devices[8:16:2] +
                            devices[9:16:2])
    # The device order in the below mesh is:
    #   [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
    # each having the following process index:
    #   The process-gpu mapping is random: @sudhakarsingh27 to figure out why so
    # and the data is:
    #   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    global_mesh = jax.sharding.Mesh(mesh_devices, ("x",))
    global_input_shape = (16,)
    mesh_axes = jax.sharding.PartitionSpec("x")
    global_input_data = np.arange(
        util.prod(global_input_shape)).reshape(global_input_shape)

    def cb(index):
      return global_input_data[index]

    gda1 = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes), cb)

    # device_id -> (index, replica_id)
    expected_idx_rid = {
        0: ((slice(0, 1),), 0),
        1: ((slice(4, 5),), 0),
        2: ((slice(1, 2),), 0),
        3: ((slice(5, 6),), 0),
        4: ((slice(2, 3),), 0),
        5: ((slice(6, 7),), 0),
        6: ((slice(3, 4),), 0),
        7: ((slice(7, 8),), 0),
        8: ((slice(8, 9),), 0),
        9: ((slice(12, 13),), 0),
        10: ((slice(9, 10),), 0),
        11: ((slice(13, 14),), 0),
        12: ((slice(10, 11),), 0),
        13: ((slice(14, 15),), 0),
        14: ((slice(11, 12),), 0),
        15: ((slice(15, 16),), 0),
    }

    with jax.sharding.Mesh(global_mesh.devices, global_mesh.axis_names):
      f = pjit.pjit(lambda x: x, out_shardings=mesh_axes)
      out = f(gda1)
      for s in out.addressable_shards:
        device_id = s.device.id
        expected_index = expected_idx_rid[device_id][0]
        expected_replica_id = expected_idx_rid[device_id][1]
        self.assertEqual(s.index, expected_index)
        self.assertEqual(s.replica_id, expected_replica_id)
        self.assertEqual(s.data.shape, (1,))
        np.testing.assert_array_equal(np.asarray(s.data),
                                      global_input_data[expected_index])

  def test_pjit_gda_non_contiguous_mesh_2d(self):
    jax.distributed.initialize()
    global_mesh = self.create_2d_non_contiguous_mesh()
    global_input_shape = (16, 2)
    mesh_axes = jax.sharding.PartitionSpec("x", "y")
    global_input_data = np.arange(
        util.prod(global_input_shape)).reshape(global_input_shape)

    def cb(index):
      return global_input_data[index]

    gda1 = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes), cb)

    # device_id -> (index, replica_id)
    expected_idx_rid = {
        0: ((slice(0, 2), slice(0, 1)), 0),
        1: ((slice(4, 6), slice(0, 1)), 0),
        2: ((slice(0, 2), slice(1, 2)), 0),
        3: ((slice(4, 6), slice(1, 2)), 0),
        4: ((slice(2, 4), slice(0, 1)), 0),
        5: ((slice(6, 8), slice(0, 1)), 0),
        6: ((slice(2, 4), slice(1, 2)), 0),
        7: ((slice(6, 8), slice(1, 2)), 0),
        8: ((slice(8, 10), slice(0, 1)), 0),
        9: ((slice(12, 14), slice(0, 1)), 0),
        10: ((slice(8, 10), slice(1, 2)), 0),
        11: ((slice(12, 14), slice(1, 2)), 0),
        12: ((slice(10, 12), slice(0, 1)), 0),
        13: ((slice(14, 16), slice(0, 1)), 0),
        14: ((slice(10, 12), slice(1, 2)), 0),
        15: ((slice(14, 16), slice(1, 2)), 0),
    }

    with global_mesh:
      f = pjit.pjit(lambda x: x, out_shardings=mesh_axes)
      out = f(gda1)

      for s in out.addressable_shards:
        device_id = s.device.id
        expected_index = expected_idx_rid[device_id][0]
        expected_replica_id = expected_idx_rid[device_id][1]
        self.assertEqual(s.index, expected_index)
        self.assertEqual(s.replica_id, expected_replica_id)
        self.assertEqual(s.data.shape, (2, 1))
        np.testing.assert_array_equal(np.asarray(s.data),
                                      global_input_data[expected_index])

    with global_mesh:
      f = pjit.pjit(
          lambda x: x,
          in_shardings=jax.sharding.PartitionSpec(None),
          out_shardings=mesh_axes,
      )
      # Fully replicated values allows a non-contiguous mesh.
      out = f(global_input_data)

    with global_mesh:
      f = pjit.pjit(lambda x: x, in_shardings=None, out_shardings=mesh_axes)
      # Fully replicated values allows a non-contiguous mesh.
      out = f(global_input_data)

    gda2 = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec(None)), cb)

    with global_mesh:
      f = pjit.pjit(
          lambda x, y: (x, y),
          in_shardings=(None, None),
          out_shardings=(mesh_axes, mesh_axes),
      )
      # Fully replicated values + GDA allows a non-contiguous mesh.
      out1, out2 = f(global_input_data, gda2)

  def test_pjit_gda_non_contiguous_mesh_2d_aot(self):
    jax.distributed.initialize()
    global_mesh = self.create_2d_non_contiguous_mesh()
    global_input_shape = (8, 2)
    mesh_axes = jax.sharding.PartitionSpec("x", "y")
    global_input_data = np.arange(
        util.prod(global_input_shape)).reshape(global_input_shape)
    gda1 = jax.make_array_from_callback(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes),
        lambda idx: global_input_data[idx])

    with global_mesh:
      f = pjit.pjit(
          lambda x, y: (x, y),
          in_shardings=jax.sharding.PartitionSpec("x", "y"),
          out_shardings=jax.sharding.PartitionSpec("x", "y"),
      )
      inp_aval = core.ShapedArray((8, 2), jnp.int32)
      # `ShapedArray` is considered global when lowered and compiled.
      # Hence it can bypass the contiguous mesh restriction.
      compiled = f.lower(inp_aval, gda1).compile()
      out1, out2 = compiled(gda1, gda1)
      self.assertEqual(out1.shape, (8, 2))
      self.assertEqual(out2.shape, (8, 2))

  def test_pjit_gda_eval_shape(self):
    jax.distributed.initialize()

    with jtu.create_global_mesh((16,), ("x")):

      @functools.partial(pjit.pjit,
                         in_shardings=jax.sharding.PartitionSpec(None),
                         out_shardings=jax.sharding.PartitionSpec("x"))
      def f():
        return jnp.zeros([32, 10])

      self.assertEqual(f().shape, (32, 10))
      self.assertEqual(jax.eval_shape(f).shape, (32, 10))

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
