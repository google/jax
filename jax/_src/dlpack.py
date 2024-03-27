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
from typing import Any, Optional
import warnings

from jax._src.api import device_put
from jax import numpy as jnp
from jax._src import array
from jax._src import xla_bridge
from jax._src.lib import xla_client
from jax._src.lib import xla_extension_version
from jax._src.typing import Array
from jax._src.sharding import Sharding

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


def to_dlpack(x: Array, take_ownership: bool = False,
              stream: int | Any | None = None):
  """Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``.

  Args:
    x: a :class:`~jax.Array`, on either CPU or GPU.
    take_ownership: Deprecated. It is a no-op to set take_ownership. Will be
      deleted in 01/2024.
    stream: optional platform-dependent stream to wait on until the buffer is
      ready. This corresponds to the `stream` argument to ``__dlpack__``
      documented in https://dmlc.github.io/dlpack/latest/python_spec.html.

  Returns:
    A dlpack PyCapsule object.

  Note:
    While JAX arrays are always immutable, dlpack buffers cannot be marked as
    immutable, and it is possible for processes external to JAX to mutate them
    in-place. If a dlpack buffer derived from a JAX array is mutated, it may
    lead to undefined behavior when using the associated JAX array.
  """
  if not isinstance(x, array.ArrayImpl):
    raise TypeError("Argument to to_dlpack must be a jax.Array, "
                    f"got {type(x)}")
  assert len(x.devices()) == 1
  if take_ownership:
    warnings.warn(
        "take_ownership in to_dlpack is deprecated and it is a no-op."
    )
  return xla_client._xla.buffer_to_dlpack_managed_tensor(
      x.addressable_data(0), stream=stream
  )  # type: ignore

def _place_array(_arr, device, dlpack_device, copy):
  if device and dlpack_device != device:
    if copy is not None and not copy:
      raise ValueError(
        f"Specified {device=} which requires a copy since the source device "
        f"is {repr(dlpack_device)}, however copy=False. Set copy=True or "
        "copy=None to perform the requested operation."
      )
    else:
      return device_put(_arr, device)
  if copy:
    return jnp.array(_arr, copy=True)
  return _arr

def _legacy_from_dlpack(dlpack, device: xla_client.Device | None = None , copy: Optional[bool] = None):
  preferred_platform = getattr(device, "platform", None)
  if device and preferred_platform == "gpu":
    preferred_platform = "cuda" if "cuda" in device.client.platform_version else "rocm"

  cpu_backend = xla_bridge.get_backend("cpu")
  gpu_backend = None

  if preferred_platform in {"cuda", "rocm"}:
    try:
      gpu_backend = xla_bridge.get_backend(preferred_platform)
    except RuntimeError:
      raise TypeError(
        f"A {str.upper(preferred_platform)} device was specified, however no "
        f"{str.upper(preferred_platform)} backend was found."
      )

  if preferred_platform is None:
    try:
      gpu_backend = xla_bridge.get_backend("cuda")
    except RuntimeError:
      pass
    # Try ROCm if CUDA backend not found
    if gpu_backend is None:
      try:
        gpu_backend = xla_bridge.get_backend("rocm")
      except RuntimeError:
        pass

  _arr = jnp.asarray(xla_client._xla.dlpack_managed_tensor_to_buffer(
      dlpack, cpu_backend, gpu_backend)) # type: ignore

  return _place_array(_arr, device, _arr.devices().pop(), copy)

def _from_dlpack(external_array, device: xla_client.Device | None = None , copy: bool | None = None):
  dl_device_type, device_id = external_array.__dlpack_device__()
  try:
    dl_device_platform = {
        DLDeviceType.kDLCPU: "cpu",
        DLDeviceType.kDLCUDA: "cuda",
        DLDeviceType.kDLROCM: "rocm",
    }[dl_device_type]
  except TypeError:
    # https://dmlc.github.io/dlpack/latest/python_spec.html recommends using
    # TypeError.
    raise TypeError(
        "Array passed to from_dlpack is on unsupported device type "
        f"(DLDeviceType: {dl_device_type}, array: {external_array}")

  backend = xla_bridge.get_backend(dl_device_platform)
  dlpack_device = backend.device_from_local_hardware_id(device_id)
  try:
    stream = dlpack_device.get_stream_for_external_ready_events()
  except xla_client.XlaRuntimeError as err:  # type: ignore
    if "UNIMPLEMENTED" in str(err):
      stream = None
    else:
      raise
  dlpack = external_array.__dlpack__(stream=stream)

  _arr = jnp.asarray(xla_client._xla.dlpack_managed_tensor_to_buffer(
      dlpack, dlpack_device, stream))

  return _place_array(_arr, device, dlpack_device, copy)

def from_dlpack(external_array, device: xla_client.Device | Sharding | None = None , copy: bool | None = None):
  """Returns a :class:`~jax.Array` representation of a DLPack tensor.

  The returned :class:`~jax.Array` shares memory with ``external_array`` if no
  device transfer or copy was requested.

  Args:
    external_array: An array object that has __dlpack__ and __dlpack_device__
      methods, or a DLPack tensor on either CPU or GPU (legacy API).

    device: The (optional) :py:class:`Device`, representing the device on which
      the returned array should be placed. If given, then the result is committed
      to the device. If unspecified, the resulting array will be unpacked onto the
      same device it originated from. Setting ``device`` to a device different from
      the source of ``external_array`` will require a copy, meaning ``copy`` must be
      set to either ``True`` or ``None``.

    copy: An (optional) boolean, controlling whether or not to a copy is performed.
      If ``copy=True`` then a copy is always performed, even if unpacked onto the
      same device. If ``copy=False`` then the copy is never peformed and will raise
      an error if necessary. When ``copy=None`` then a copy may be performed if
      needed for a device transfer.

  Returns:
    A jax.Array

  Note:
    While JAX arrays are always immutable, dlpack buffers cannot be marked as
    immutable, and it is possible for processes external to JAX to mutate them
    in-place. If a jax Array is constructed from a dlpack buffer and the buffer
    is later modified in-place, it may lead to undefined behavior when using
    the associated JAX array.
  """
  if isinstance(device, Sharding):
    device_set = device.device_set
    if len(device_set) > 1:
      raise ValueError(
        "from_dlpack can only unpack a dlpack tensor onto a singular device, but "
        f"a Sharding with {len(device_set)} devices was provided."
      )
    device = device_set.pop()
  if hasattr(external_array, "__dlpack__"):
    return _from_dlpack(external_array, device, copy)

  # Legacy path
  return _legacy_from_dlpack(external_array, device, copy)
