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

from __future__ import annotations

from jax._src import clusters
import socket

from numpy import unique, where

from importlib.util import find_spec

class Mpi4pyCluster(clusters.ClusterEnv):
  
  @classmethod
  def is_env_present(cls, opt_in=False) -> bool:
    # Why include and opt_in?  Enables this class to conform to
    # every other ClusterEnv subclass while always being rejected
    # as viable, except in the express case where we request to check
    # it explicitly.

    # in many HPC clusters, the variables `https_proxy` and `http_proxy`
    # are set to enable access to normally unreachable network locations.
    # For example, `pip install ...` fails on compute nodes without them.

    # Unfortunately, these variables break the jax distributed init.
    # The user needs to unset them, but I don't want to modify the global
    # python os.environ here for them (bad practice)

    # And I also don't know what the right way to raise a complaint here is
    
    # Relies on mpi4py:
    return find_spec("mpi4py") is not None and opt_in == True

  @classmethod
  def get_coordinator_address(cls) -> str:

    # Using mpi4py, figure out rank 0 and it's hostname.
    # Then broadcast the hostname and port.


    from mpi4py import MPI
    # Get the global communicator:
    COMM_WORLD = MPI.COMM_WORLD

    # On rank 0, get the hostname:

    if COMM_WORLD.Get_rank() == 0:
        # Order all the hostnames, and find unique ones
        hostname = socket.gethostname()

        # Apparently, we want to pick a port in an ephemeral range...
        port_id = hash(hostname) % 2**12 + (65535 - 2**12 + 1)
        
        hostname = f'{hostname}:{port_id}'

    else:
        hostname = None
   


    # Broadcast the host_ip to all ranks:
    hostname = COMM_WORLD.bcast(hostname, root=0)
    # host_ip = COMM_WORLD.bcast(host_ip, root=0)


    return hostname


  @classmethod
  def get_process_count(cls) -> int:
    from mpi4py import MPI
    return int(MPI.COMM_WORLD.Get_size())

  @classmethod
  def get_process_id(cls) -> int:
    from mpi4py import MPI
    return int(MPI.COMM_WORLD.Get_rank())
  
  @classmethod
  def get_local_process_id(cls) -> int | None:

    # Using mpi4py, split the global communicator into sub communicators
    # based on hostname.  mpi will assign them ranks and that will allow
    # a selection of the local process ID.
    from mpi4py import MPI
    COMM_WORLD = MPI.COMM_WORLD

    hostname = socket.gethostname()
    # host_key = host_key %
    all_hostnames = COMM_WORLD.gather(hostname, root=0)

    if COMM_WORLD.Get_rank() == 0:
        # Order all the hostnames, and find unique ones
        unique_hosts = unique(all_hostnames)
        # Numpy automatically sorts them.
    else:
        unique_hosts = None

    # Broadcast the list of hostnames:
    unique_hosts = COMM_WORLD.bcast(unique_hosts, root=0)

    # Find the integer for this host in the list of hosts:
    i = int(where(unique_hosts == hostname)[0])

    new_comm = COMM_WORLD.Split(color=i)


    # The rank in the new communicator - which is host-local only - IS the local rank:
    return int(new_comm.Get_rank())
