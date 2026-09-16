# Copyright 2017 The JAX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An XLA client in Python."""

from __future__ import annotations

import atexit
from collections.abc import Mapping
import contextlib
import enum
import logging
import os
import threading
from typing import Any, Protocol

from jaxlib import _jax as _xla

from jaxlib import _hlo as hlo

# Note this module does *not* depend on any Python protocol buffers. The XLA
# Python bindings are currently packaged both as part of jaxlib and as part
# of TensorFlow. If we use protocol buffers here, then importing both jaxlib
# and TensorFlow may fail with duplicate protocol buffer message definitions.

# Most functions are snake_case for consistency with other modules, some
# method names are CamelCase for consistency with XLA.

ifrt_programs = _xla.ifrt_programs

# Just an internal arbitrary increasing number to help with backward-compatible
# changes. In JAX, reference this via jax._src.lib.jaxlib_extension_version.
#
# Please suffix the version number with a brief description of your change
# in a comment. The goal here is to force a merge conflict if two changes
# attempt to grab the same version number.
_version = 495  # Non-addressable shards in batched_copy_to_host_async

# An internal increasing version number for protecting jaxlib code against
# ifrt changes.
# lives in xla/python/version.h.
# In JAX, reference this via jax._src.lib.ifrt_version.
_ifrt_version = _xla.ifrt_version_number

xla_platform_names = {
    'cpu': 'Host',
    'gpu': 'CUDA',
}

logger = logging.getLogger(__name__)

_NameValueMapping = Mapping[str, str | int | list[int] | float | bool]


def make_cpu_client(
    asynchronous=True,
    distributed_client=None,
    node_id=0,
    num_nodes=1,
    collectives=None,
    num_devices=None,
    get_local_topology_timeout_minutes=None,
    get_global_topology_timeout_minutes=None,
    transfer_server_factory=None,
) -> Client:
  register_custom_call_handler('cpu', _xla.register_custom_call_target)
  register_custom_type_handler('cpu', _xla.register_custom_type)
  return _xla.get_tfrt_cpu_client(
      asynchronous=asynchronous,
      distributed_client=distributed_client,
      node_id=node_id,
      num_nodes=num_nodes,
      collectives=collectives,
      num_devices=num_devices,
      get_local_topology_timeout_minutes=get_local_topology_timeout_minutes,
      get_global_topology_timeout_minutes=get_global_topology_timeout_minutes,
      transfer_server_factory=transfer_server_factory,
  )


DeviceTopology = _xla.DeviceTopology
get_topology_for_devices = _xla.get_topology_for_devices


def make_tfrt_tpu_c_api_device_topology(
    topology_name: str = '', **kwargs
) -> DeviceTopology:
  """Creates a PJRT C API TopologyDescription."""
  return _xla.get_default_c_api_topology('tpu', topology_name, dict(**kwargs))


def make_c_api_device_topology(
    c_api: Any, topology_name: str = '', **kwargs
) -> DeviceTopology:
  """Creates a PJRT C API TopologyDescription."""
  return _xla.get_c_api_topology(c_api, topology_name, dict(**kwargs))


def pjrt_plugin_loaded(plugin_name: str) -> bool:
  return _xla.pjrt_plugin_loaded(plugin_name)


def load_pjrt_plugin_dynamically(plugin_name: str, library_path: str) -> Any:
  return _xla.load_pjrt_plugin(plugin_name, library_path, c_api=None)


def load_pjrt_plugin_with_c_api(plugin_name: str, c_api: Any) -> None:
  _xla.load_pjrt_plugin(plugin_name, None, c_api)


def pjrt_plugin_initialized(plugin_name: str) -> bool:
  return _xla.pjrt_plugin_initialized(plugin_name)


def initialize_pjrt_plugin(plugin_name: str) -> None:
  """Initializes a PJRT plugin.

  The plugin needs to be loaded first (through load_pjrt_plugin_dynamically or
  static linking) before this method is called.
  Args:
    plugin_name: the name of the PJRT plugin.
  """
  _xla.initialize_pjrt_plugin(plugin_name)


def make_c_api_client(
    plugin_name: str,
    options: _NameValueMapping | None = None,
    distributed_client: _xla.DistributedRuntimeClient | None = None,
    transfer_server_factory: _xla.TransferServerInterfaceFactory | None = None,
    force_dcn_cross_host_transfers: bool = False,
    sort_devices_by_process_index: bool = True,
):
  """Creates a PJRT C API client for a PJRT plugin.

  It is required that load_pjrt_plugin_dynamically is called once with the same
  plugin_name before this method is called.

  Args:
     plugin_name: the name of the PJRT plugin.
     options: extra platform-specific options.
     distributed_client: distributed client.

  Returns:
     A PJRT C API client for plugin_name.
  """
  if options is None:
    options = {}
  return _xla.get_c_api_client(
      plugin_name,
      options,
      distributed_client,
      transfer_server_factory,
      force_dcn_cross_host_transfers,
      sort_devices_by_process_index,
  )


def generate_pjrt_gpu_plugin_options() -> _NameValueMapping:
  """Generates the PjRt GPU plugin options.

  Returns:
    A dictionary of plugin options.
  """

  options: dict[str, Any] = {}
  options['platform_name'] = 'cuda'
  allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default').lower()
  memory_fraction = os.getenv('XLA_CLIENT_MEM_FRACTION', '')
  deprecated_memory_fraction = os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION', '')
  if deprecated_memory_fraction:
    if memory_fraction:
      raise ValueError(
          'XLA_CLIENT_MEM_FRACTION is specified together '
          'with XLA_PYTHON_CLIENT_MEM_FRACTION. '
          'Remove the latter one, it is deprecated.'
      )
    else:
      memory_fraction = deprecated_memory_fraction
  preallocate = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE', '')
  collective_memory_size = os.getenv(
      'XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB', ''
  )
  allowed_allocators = (
      'default',
      'platform',
      'bfc',
      'cuda_async',
      'vmm',
      'address',
  )
  if allocator not in allowed_allocators:
    raise ValueError(
        'XLA_PYTHON_CLIENT_ALLOCATOR env var must be one of %s, got "%s"'
        % (', '.join(f'"{a}"' for a in allowed_allocators), allocator)
    )
  options['allocator'] = allocator
  if memory_fraction:
    options['memory_fraction'] = float(memory_fraction)
  if preallocate:
    options['preallocate'] = preallocate not in ('false', 'False', '0')
  if collective_memory_size:
    options['collective_memory_size'] = int(collective_memory_size) * (1 << 20)
  abort = os.getenv('XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE', '0')
  options['abort_collectives_on_failure'] = bool(int(abort))
  use_trft_gpu_client = os.getenv('XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT', '0')
  options['use_tfrt_gpu_client'] = bool(int(use_trft_gpu_client))
  return options


PrimitiveType = hlo.PrimitiveType

Shape = hlo.Shape
Shape.__doc__ = """
A Shape is an object defined in C++ that duck types like the following class:

class Shape:
  '''Represents an XLA shape.

  A shape is either an array shape, having rank-many integer
  dimensions and an element type (represented by a Numpy dtype), or it
  is a tuple shape, having a shape for every tuple component:

    type shape =
        TupleShape of shape list
      | ArrayShape of { dimensions: int list; element_type: dtype }
  '''

  @staticmethod
  def tuple_shape(tuple_shapes) -> Shape:
    "Construct a tuple shape."

  @staticmethod
  def array_shape(element_type, dimensions, minor_to_major=None) -> Shape:

  @staticmethod
  def from_pyval(pyval) -> Shape:
    "Returns a Shape that describes a tuple-tree of Numpy arrays."

  def __init__(self, str) -> Shape:
    "Parses a shape string."
  def __eq__(self, other: Shape) -> bool:
  def __ne__(self, other: Shape) -> bool:
  def __hash__(self):
  def __repr__(self):
  def is_tuple(self) -> bool:
  def is_array(self) -> bool:
  def tuple_shapes(self) -> [Shape]:
  def numpy_dtype(self) -> np.dtype:
    "Like element_type(), but returns dtype('O') for a tuple shape."
  def xla_element_type(self) -> PrimitiveType:
  def element_type(self) -> np.dtype:
  def dimensions(self) -> (int, int, ...):
  def rank(self) -> int:
  def with_major_to_minor_layout_if_absent(self) -> Shape:
    "Returns a copy with missing layouts set to major-to-minor."

  def to_serialized_proto(self) -> bytes:
    "Returns 'shape' as a serialized proto."
"""

ProgramShape = hlo.ProgramShape
ProgramShape.__doc__ = """
A ProgramShape is a C++ object that duck types like the following class.

class ProgramShape:
  def __init__(self, parameter_shapes, result_shape):
  def parameter_shapes(self) -> [Shape]:
  def result_shape(self) -> Shape:
  def __repr__(self):
"""

DeviceAssignment = _xla.DeviceAssignment
DeviceAssignment.__doc__ = """
A DeviceAssignment is a C++ object with the following signature.

def create(assignment):
  '''Builds a device assignment.

   Args:
     assignment: a 2D numpy array of device ordinal integers, indexed by
       [replica][computation_in_replica].
   Returns:
     A device assignment.
  '''

def replica_count():
  '''Returns the number of replicas.'''
def computation_count():
  '''Returns the number of computations per replica.'''
"""

Device = _xla.Device
CompileOptions = _xla.CompileOptions

HostBufferSemantics = _xla.HostBufferSemantics

# An Executable is a C++ class that duck types with the following API:
# class Executable:
#   def local_devices(self) -> [Device]:
#   def execute(self, arguments : [Buffer]) -> Buffer:
#     """Execute on one replica with Buffer arguments and return value."""
#
#   def size_of_generated_code_in_bytes(self) -> int:
#     """Return generated binary size, or -1 if not known."""
#
#   def execute_sharded_on_local_devices(self, arguments: [[Buffer]])
#       -> [Buffer]:
#     """Execute on many replicas with Buffer arguments and return value.
#
#     Args:
#       arguments: A sequence of sequences of Buffers. The i'th element of each
#         sequence comprises the arguments for execution on the i'th local
#         device.
#
#     Returns:
#       A list of the computation's outputs as a list of Buffers for each
#       device.
#     """
#
# There are different implementations of Executable for different backends.


XlaComputation = hlo.XlaComputation
Client = _xla.Client
Memory = _xla.Memory
Array = _xla.Array
ArrayImpl = _xla.ArrayImpl
LoadedExecutable = _xla.LoadedExecutable
Executable = _xla.Executable
DeviceList = _xla.DeviceList
OpSharding = hlo.OpSharding
HloSharding = hlo.HloSharding
Sharding = _xla.Sharding
NamedSharding = _xla.NamedSharding
SingleDeviceSharding = _xla.SingleDeviceSharding
GSPMDSharding = _xla.GSPMDSharding
PjRtLayout = _xla.PjRtLayout
AutotuneCacheMode = _xla.AutotuneCacheMode


def LoadedExecutable_execute(self, arguments, device=None):
  del device
  results = self.execute_sharded(arguments)
  return [x[0] for x in results.disassemble_into_single_device_arrays()]


def LoadedExecutable_execute_with_token(self, arguments, device=None):
  del device
  results = self.execute_sharded(arguments, with_tokens=True)
  return (
      [x[0] for x in results.disassemble_into_single_device_arrays()],
      results.consume_token().get_token(0),
  )


LoadedExecutable.execute = LoadedExecutable_execute
LoadedExecutable.execute_with_token = LoadedExecutable_execute_with_token


class CustomCallTargetTraits(enum.IntFlag):
  DEFAULT = 0
  # Calls to custom call are safe to trace into the command buffer. It means
  # that calls to custom call always launch exactly the same device operations
  # (can depend on attribute values) that can be captured and then replayed.
  #
  # Supported only for custom calls implemented with XLA FFI.
  COMMAND_BUFFER_COMPATIBLE = 1


class CustomCallHandler(Protocol):

  def __call__(
      self,
      name: str,
      fn: Any,
      platform: str,
      /,
      api_version: int = ...,
      traits: CustomCallTargetTraits = ...,
  ) -> None:
    ...


_custom_callback_handler: dict[str, CustomCallHandler] = {}
# Key is xla_platform_name, value is (function_name, function, api_version)
_custom_callback: dict[
    str, list[tuple[str, Any, int, CustomCallTargetTraits]]
] = {}
_custom_callback_lock = threading.Lock()


def register_custom_call_target(
    name: str,
    fn: Any,
    platform: str = 'cpu',
    api_version: int = 0,
    traits: CustomCallTargetTraits = CustomCallTargetTraits.DEFAULT,
) -> None:
  """Registers a custom call target.

  Args:
    name: bytes containing the name of the function.
    fn: a PyCapsule object containing the function pointer.
    platform: the target platform.
    api_version: the XLA FFI version to use. Supported versions are: 0 for the
      untyped FFI and 1 for the typed FFI.
    traits: custom call traits corresponding to XLA FFI handler traits.
  """
  # To support AMD GPUs, we need to have xla_platform_names["gpu"] == "ROCM"
  # Since that is hardcoded to CUDA, we are using the following as workaround.
  xla_platform_name = xla_platform_names.get(platform, platform)
  with _custom_callback_lock:
    if xla_platform_name in _custom_callback_handler:
      _custom_callback_handler[xla_platform_name](
          name, fn, xla_platform_name, api_version, traits
      )
    else:
      _custom_callback.setdefault(xla_platform_name, []).append(
          (name, fn, api_version, traits)
      )


def register_custom_call_handler(
    platform: str, handler: CustomCallHandler
) -> None:
  """Registers a custom handler and use it to register existing custom calls.

  If a custom call handler for the platform already exist, calling this method
  is a no-op and it will not register a new handler.

  Args:
    platform: the target platform.
    handler: the function to register a custom call.
  """
  xla_platform_name = xla_platform_names.get(platform, platform)
  with _custom_callback_lock:
    if xla_platform_name in _custom_callback_handler:
      logger.debug(
          'Custom call handler for %s is already register. Will not register a'
          ' new one',
          xla_platform_name,
      )
      return
    _custom_callback_handler[xla_platform_name] = handler
    if xla_platform_name in _custom_callback:
      for name, fn, api_version, traits in _custom_callback[xla_platform_name]:
        handler(name, fn, xla_platform_name, api_version, traits)
      del _custom_callback[xla_platform_name]


class CustomTypeIdHandler(Protocol):

  def __call__(self, type_name: str, type_id: Any, /) -> None:
    ...


_custom_type_id_handler: dict[str, CustomTypeIdHandler] = {}
_custom_type_id: dict[str, Any] = {}
_custom_type_id_lock = threading.Lock()


def register_custom_type(
    type_name: str,
    type_id: Any,
    platform: str = 'cpu',
) -> None:
  """Register a custom type id for use with the FFI.

  Args:
    type_name: a unique name for the type.
    type_id: a PyCapsule object containing a pointer to the ``ffi::TypeId``.
    platform: the target platform.
  """
  xla_platform_name = xla_platform_names.get(platform, platform)
  with _custom_type_id_lock:
    if xla_platform_name in _custom_type_id_handler:
      _custom_type_id_handler[xla_platform_name](type_name, type_id)
    else:
      _custom_type_id.setdefault(xla_platform_name, []).append(
          (type_name, type_id)
      )


def register_custom_type_handler(
    platform: str, handler: CustomTypeIdHandler
) -> None:
  """Register a custom type id handler and use it to register existing type ids.

  If a custom type id handler for the platform already exist, calling this
  method is a no-op and it will not register a new handler.

  Args:
    platform: the target platform.
    handler: the function to register a custom type id.
  """
  xla_platform_name = xla_platform_names.get(platform, platform)
  with _custom_callback_lock:
    if xla_platform_name in _custom_type_id_handler:
      logger.debug(
          'Custom type id handler for %s is already register. Will not '
          'register a new one',
          xla_platform_name,
      )
      return
    _custom_type_id_handler[xla_platform_name] = handler
    if xla_platform_name in _custom_type_id:
      for name, capsule in _custom_type_id[xla_platform_name]:
        handler(name, capsule)
      del _custom_type_id[xla_platform_name]


register_custom_call_partitioner = _xla.register_custom_call_partitioner
encode_inspect_sharding_callback = _xla.encode_inspect_sharding_callback
hlo_sharding_util = _xla.hlo_sharding_util
register_custom_call_as_batch_partitionable = (
    _xla.register_custom_call_as_batch_partitionable
)


Traceback = _xla.Traceback
TracebackScope = _xla.TracebackScope
Frame = _xla.Frame


@contextlib.contextmanager
def execution_stream_id(new_id: int):
  """Context manager that overwrites and restores the current thread's execution_stream_id."""
  saved = _xla.get_execution_stream_id()
  _xla.set_execution_stream_id(new_id)
  try:
    yield
  finally:
    _xla.set_execution_stream_id(saved)


XlaRuntimeError = _xla.JaxRuntimeError

# Perform one last garbage collection of deferred Python references. This is
# mostly to keep ASAN happy.
atexit.register(_xla.collect_garbage)

array_result_handler = _xla.array_result_handler
batched_copy_array_to_devices_with_sharding = (
    _xla.batched_copy_array_to_devices_with_sharding
)
batched_device_put = _xla.batched_device_put
batched_copy_to_host_async = _xla.batched_copy_to_host_async
reorder_shards = _xla.reorder_shards
batched_block_until_ready = _xla.batched_block_until_ready
check_and_canonicalize_memory_kind = _xla.check_and_canonicalize_memory_kind
Layout = hlo.Layout
custom_call_targets = _xla.custom_call_targets
ArrayCopySemantics = _xla.ArrayCopySemantics
