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

from __future__ import annotations

import enum
from typing import Any
import warnings

from jax import numpy as jnp
from jax._src import array
from jax._src import xla_bridge
from jax._src.lax.lax import _array_copy
from jax._src.lib import xla_client
from jax._src.lib import xla_extension_version
from jax._src.typing import Array
from jax._src.api import device_put

DLPACK_VERSION = (0, 8)
MIN_DLPACK_VERSION = (0, 5)

# A set of dtypes that dlpack supports.
# Note: Make sure to use a "type", not a dtype instance, when looking up this set
# because their hashes are different.
# For example,
# hash(jnp.float32) != hash(jnp.dtype(jnp.float32))
# hash(jnp.float32) == hash(jnp.dtype(jnp.float32).type)
# TODO(phawkins): Migrate to using dtypes instead of the scalar type objects.
SUPPORTED_DTYPES = frozenset({
    jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16,
    jnp.uint32, jnp.uint64, jnp.float16, jnp.bfloat16, jnp.float32,
    jnp.float64, jnp.complex64, jnp.complex128})

if xla_extension_version >= 231:
  SUPPORTED_DTYPES = SUPPORTED_DTYPES | frozenset({jnp.bool_})


# Mirror of dlpack.h enum
class DLDeviceType(enum.IntEnum):
  kDLCPU = 1
  kDLCUDA = 2
  kDLROCM = 10

def _to_dlpack(x: Array, stream: int | Any | None,
               src_device: xla_client.Device | None = None,
               device: xla_client.Device | None = None,
               copy: bool | None = None):

  if src_device is None:
    src_device, = x.devices()
  if device and (src_device is None or device != src_device):
    if copy is not None and not copy:
      raise ValueError(
        f"Specified {device=} which requires a copy since the source device "
        f"is {repr(src_device)}, however copy=False. Set copy=True or "
        "copy=None to perform the requested operation."
      )
    else:
      arr = device_put(x, device)
  else:
    arr = _array_copy(x) if copy else x
  return xla_client._xla.buffer_to_dlpack_managed_tensor(
    arr.addressable_data(0), stream=stream
  )

def to_dlpack(x: Array, take_ownership: bool = False,
              stream: int | Any | None = None,
              src_device: xla_client.Device | None = None,
              dl_device: tuple[DLDeviceType, int] | None = None,
              max_version: tuple[int, int] | None = None,
              copy : bool | None = None):
  """Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``.

  Args:
    x: a :class:`~jax.Array`, on either CPU or GPU.
    take_ownership: Deprecated. It is a no-op to set take_ownership. Will be
      deleted in 01/2024.
    stream: optional platform-dependent stream to wait on until the buffer is
      ready. This corresponds to the `stream` argument to ``__dlpack__``
      documented in https://dmlc.github.io/dlpack/latest/python_spec.html.
    src_device: either a CPU or GPU :class:`~jax.Device`.
    dl_device: a tuple of ``(dl_device_type, local_hardware_id)`` in DLPack
      format e.g. as produced by ``__dlpack_device__``.
    max_version: the maximum DLPack version that the consumer (i.e. caller of
      ``__dlpack__``) supports in the form of a 2-tuple of ``(major, minor)``.
      This function is not guaranteed to return a capsule of version
      ``max_version``.
    copy: a boolean indicating whether or not to copy the input. If
      ``copy=True`` then the function must always copy. When
      ``copy=False`` then the function must never copy, and must raise an error
      when a copy is deemed necessary. If ``copy=None`` then the function must
      avoid a copy if possible but may copy if needed.

  Returns:
    A DLPack PyCapsule object.

  Note:
    While JAX arrays are always immutable, ``DLPackManagedTensor`` buffers
    cannot be marked as immutable, and it is possible for processes external
    to JAX to mutate them in-place. If a DLPack buffer derived from a JAX array
    is mutated, it may lead to undefined behavior when using the associated JAX
    array. When JAX eventually supports ``DLManagedTensorVersioned``
    (DLPack 1.0), it will be possible to specify that a buffer is read-only.
  """
  if not isinstance(x, array.ArrayImpl):
    raise TypeError("Argument to to_dlpack must be a jax.Array, "
                    f"got {type(x)}")
  if take_ownership:
    warnings.warn(
        "take_ownership in to_dlpack is deprecated and it is a no-op."
    )

  device = None
  dl_device_type, local_hardware_id = dl_device if dl_device else (None, None)
  if dl_device_type:
    try:
      dl_device_platform = {
          DLDeviceType.kDLCPU: "cpu",
          DLDeviceType.kDLCUDA: "cuda",
          DLDeviceType.kDLROCM: "rocm",
      }[dl_device_type]
      backend = xla_bridge.get_backend(dl_device_platform)
      device = backend.device_from_local_hardware_id(local_hardware_id)
    except TypeError:
      # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
      # recommends using BufferError.
      raise BufferError(
          "The device specification passed to to_dlpack contains an unsupported "
          f"device type (DLDeviceType: {dl_device_type})")

  # As new versions are adopted over time, we can maintain some legacy paths
  # for compatability mediated through the max_version parameter.
  # TODO(micky774): Deprecate default usage of DLPackManagedTensor when XLA
  # supports DLManagedTensorVersioned (DLPack version 1.0) and repurpose the
  # current _to_dlpack as a legacy path for (0,5) <= max_version < (1,0).
  if max_version is None or max_version >= DLPACK_VERSION:
    # Latest
    return _to_dlpack(
      x, stream=stream,
      src_device=src_device,
      device=device,
      copy=copy
    )
  elif max_version >= MIN_DLPACK_VERSION:
    # Oldest supported
    return _to_dlpack(
      x, stream=stream,
      src_device=src_device,
      device=device,
      copy=copy
    )
  else:
    raise BufferError(
      f"JAX does not support any version below {MIN_DLPACK_VERSION} but "
      f"version ({max_version}) was requested."
    )


def from_dlpack(external_array):
  """Returns a :class:`~jax.Array` representation of a DLPack tensor.

  The returned :class:`~jax.Array` shares memory with ``external_array``.

  Args:
    external_array: an array object that has __dlpack__ and __dlpack_device__
      methods, or a DLPack tensor on either CPU or GPU (legacy API).

  Returns:
    A jax.Array

  Note:
    While JAX arrays are always immutable, dlpack buffers cannot be marked as
    immutable, and it is possible for processes external to JAX to mutate them
    in-place. If a jax Array is constructed from a dlpack buffer and the buffer
    is later modified in-place, it may lead to undefined behavior when using
    the associated JAX array.
  """
  if hasattr(external_array, "__dlpack__"):
    dl_device_type, device_id = external_array.__dlpack_device__()
    try:
      device_platform = {
          DLDeviceType.kDLCPU: "cpu",
          DLDeviceType.kDLCUDA: "cuda",
          DLDeviceType.kDLROCM: "rocm",
      }[dl_device_type]
    except TypeError as err:
      # https://dmlc.github.io/dlpack/latest/python_spec.html recommends using
      # TypeError.
      raise BufferError(
          "Array passed to from_dlpack is on unsupported device type "
          f"(DLDeviceType: {dl_device_type}, array: {external_array}") from err

    backend = xla_bridge.get_backend(device_platform)
    device = backend.device_from_local_hardware_id(device_id)
    try:
      stream = device.get_stream_for_external_ready_events()
    except xla_client.XlaRuntimeError as err:  # type: ignore
      if "UNIMPLEMENTED" in str(err):
        stream = None
      else:
        raise
    dlpack = external_array.__dlpack__(stream=stream)

    return jnp.asarray(xla_client._xla.dlpack_managed_tensor_to_buffer(
        dlpack, device, stream))
  else:
    # Legacy path
    dlpack = external_array
    cpu_backend = xla_bridge.get_backend("cpu")
    try:
      gpu_backend = xla_bridge.get_backend("cuda")
    except RuntimeError:
      gpu_backend = None

    # Try ROCm if CUDA backend not found
    if gpu_backend is None:
      try:
        gpu_backend = xla_bridge.get_backend("rocm")
      except RuntimeError:
        gpu_backend = None

    return jnp.asarray(xla_client._xla.dlpack_managed_tensor_to_buffer(
        dlpack, cpu_backend, gpu_backend))
