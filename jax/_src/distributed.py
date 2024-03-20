# Copyright 2021 The JAX Authors.
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

import atexit
from collections.abc import Sequence
import logging
import os
from typing import Any

from jax._src import clusters
from jax._src import config
from jax._src import xla_bridge
from jax._src.lib import xla_extension

logger = logging.getLogger(__name__)


class State:
  process_id: int = 0
  num_processes: int = 1
  service: Any | None = None
  client: Any | None = None
  preemption_sync_manager: Any | None = None
  coordinator_address: str | None = None

  def initialize(self,
                 coordinator_address: str | None = None,
                 num_processes: int | None = None,
                 process_id: int | None = None,
                 local_device_ids: int | Sequence[int] | None = None,
                 spec_detection_method: str | None = None,
                 initialization_timeout: int = 300):
    coordinator_address = (coordinator_address or
                           os.environ.get('JAX_COORDINATOR_ADDRESS', None))
    if isinstance(local_device_ids, int):
      local_device_ids = [local_device_ids]

    if spec_detection_method is not None and spec_detection_method not in ["mpi4py"]:
      raise ValueError("spec_detection method should only be None, or \"mpi4py\".")

    (coordinator_address, num_processes, process_id, local_device_ids) = (
        clusters.ClusterEnv.auto_detect_unset_distributed_params(
            coordinator_address, num_processes, process_id, local_device_ids, spec_detection_method
        )
    )

    if coordinator_address is None:
      raise ValueError('coordinator_address should be defined.')
    if num_processes is None:
      raise ValueError('Number of processes must be defined.')
    if process_id is None:
      raise ValueError('The process id of the current process must be defined.')

    self.coordinator_address = coordinator_address

    if local_device_ids:
      visible_devices = ','.join(str(x) for x in local_device_ids) # type: ignore[union-attr]
      logger.info('JAX distributed initialized with visible devices: %s', visible_devices)
      config.update("jax_cuda_visible_devices", visible_devices)
      config.update("jax_rocm_visible_devices", visible_devices)

    self.process_id = process_id

    # Emit a warning about PROXY variables if they are in the user's env:
    proxy_vars = [ key for key in os.environ.keys() if 'proxy' in key.lower()]

    if len(proxy_vars) > 0:
      vars = " ".join(proxy_vars) + ". "
      warning = (
        f'JAX detected proxy variable(s) in the environment as distributed setup: {vars}'
        'On some systems, this may cause a hang of distributed.initialize and '
        'you may need to unset these ENV variable(s)'
      )
      logger.warning(warning)

    if process_id == 0:
      if self.service is not None:
        raise RuntimeError('distributed.initialize should only be called once.')
      logger.info('Starting JAX distributed service on %s', coordinator_address)
      self.service = xla_extension.get_distributed_runtime_service(
          coordinator_address, num_processes)

    self.num_processes = num_processes

    if self.client is not None:
      raise RuntimeError('distributed.initialize should only be called once.')

    self.client = xla_extension.get_distributed_runtime_client(
        coordinator_address, process_id, init_timeout=initialization_timeout)
    logger.info('Connecting to JAX distributed service on %s', coordinator_address)
    self.client.connect()

    self.initialize_preemption_sync_manager()

  def shutdown(self):
    if self.client:
      self.client.shutdown()
      self.client = None
    if self.service:
      self.service.shutdown()
      self.service = None
    if self.preemption_sync_manager:
      self.preemption_sync_manager = None

  def initialize_preemption_sync_manager(self):
    if self.preemption_sync_manager is not None:
      raise RuntimeError(
          'Preemption sync manager should only be initialized once.')
    self.preemption_sync_manager = (
        xla_extension.create_preemption_sync_manager())
    self.preemption_sync_manager.initialize(self.client)

global_state = State()


def initialize(coordinator_address: str | None = None,
               num_processes: int | None = None,
               process_id: int | None = None,
               local_device_ids: int | Sequence[int] | None = None,
               spec_detect_method: str | None = None,
               initialization_timeout: int = 300):
  """Initializes the JAX distributed system.

  Calling :func:`~jax.distributed.initialize` prepares JAX for execution on
  multi-host GPU and Cloud TPU. :func:`~jax.distributed.initialize` must be
  called before performing any JAX computations.

  The JAX distributed system serves a number of roles:

    * it allows JAX processes to discover each other and share topology information,
    * it performs health checking, ensuring that all processes shut down if any process dies, and
    * it is used for distributed checkpointing.

  If you are using TPU, Slurm, or Open MPI, all arguments are optional: if omitted, they
  will be chosen automatically.

  Otherwise, you must provide the ``coordinator_address``,
  ``num_processes``, and ``process_id`` arguments to :func:`~jax.distributed.initialize`.

  Please note: on some systems, particularly HPC clusters that only access external networks
  through proxy variables such as HTTP_PROXY, HTTPS_PROXY, etc., the call to
  :func`~jax.distributed.initialize` may timeout.  You may need to unset these variables
  prior to application launch.

  Args:
    coordinator_address: the IP address of process `0` and a port on which that
      process should launch a coordinator service. The choice of
      port does not matter, so long as the port is available on the coordinator
      and all processes agree on the port.
      May be ``None`` only on supported environments, in which case it will be chosen automatically.
      Note that special addresses like ``localhost`` or ``127.0.0.1`` usually mean that the program
      will bind to a local interface and are not suitable when running in a multi-host environment.
    num_processes: Number of processes. May be ``None`` only on supported environments, in
      which case it will be chosen automatically.
    process_id: The ID number of the current process. The ``process_id`` values across
      the cluster must be a dense range ``0``, ``1``, ..., ``num_processes - 1``.
      May be ``None`` only on supported environments; if ``None`` it will be chosen automatically.
    local_device_ids: Restricts the visible devices of the current process to ``local_device_ids``.
      If ``None``, defaults to all local devices being visible to the process except when processes
      are launched via Slurm and Open MPI on GPUs. In that case, it will default to a single device per process.
    spec_detect_method: An optional string to attempt to autodetect the configuration of the distributed
      run.  Available options are "mpipy" only at the moment, though more options may be available in the future.  
      Note that "mpi4py" method requires you to have a working `mpi4py` install in your environment, 
      and launch the applicatoin with an MPI-compatible job launcher such as `mpiexec`.  
      Legacy auto-detect options (OMPI, Slurm) remain enabled.
    initialization_timeout: Time period (in seconds) for which connection will
      be retried. If the initialization takes more than the timeout specified,
      the initialization will error. Defaults to 300 secs i.e. 5 mins.

  Raises:
    RuntimeError: If :func:`~jax.distributed.initialize` is called more than once.

  Example:

  Suppose there are two GPU processes, and process 0 is the designated coordinator
  with address ``10.0.0.1:1234``. To initialize the GPU cluster, run the
  following commands before anything else.

  On process 0:

  >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=0)  # doctest: +SKIP

  On process 1:

  >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=1)  # doctest: +SKIP
  """
  if xla_bridge.backends_are_initialized():
    raise RuntimeError("jax.distributed.initialize() must be called before "
                        "any JAX computations are executed.")
  global_state.initialize(coordinator_address, num_processes, process_id,
                          local_device_ids, spec_detect_method, initialization_timeout)
  atexit.register(shutdown)


def shutdown():
  """Shuts down the distributed system.

  Does nothing if the distributed system is not running."""
  global_state.shutdown()
