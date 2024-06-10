# Copyright 2024 The JAX Authors.
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
"""Colocated Python function API implementation."""

from __future__ import annotations

import dataclasses
import inspect
import random
import threading
from typing import Any, Callable, Sequence

import jax
from jax._src import api
from jax._src import tree_util
from jax._src.lib import xla_client as xc
from jax._src.traceback_util import api_boundary
from jax._src.util import wraps
from jax.experimental.colocated_python import func_backend
from jax.experimental.colocated_python.serialization import deserialize_specs, make_specs_for_serialized_specs, serialize_specs

ShapeDtypeStructTree = Any  # PyTree[ShapeDtypeStruct]


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionInfo:
  """User function wrapped by colocated_python."""

  fun: Callable[..., Any]
  fun_sourceinfo: str | None
  fun_signature: inspect.Signature | None


@dataclasses.dataclass(frozen=True, slots=True)
class Specialization:
  """Specialization for a colocated_python function."""

  in_specs_treedef: tree_util.PyTreeDef | None = None
  in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None
  out_specs_fn: Callable[..., Any] | None = None
  out_specs_treedef: tree_util.PyTreeDef | None = None
  out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None
  devices: xc.DeviceList | None = None


def _apply_specialization(
    base_specialization: Specialization,
    in_specs: ShapeDtypeStructTree | None = None,
    out_specs_fn: Callable[..., Any] | None = None,
    out_specs: ShapeDtypeStructTree | None = None,
    devices: Sequence[jax.Device] | None = None,
) -> Any:
  """Applies extra specialization to the base specialization."""
  updates: dict[str, Any] = {}

  if in_specs is not None:
    if base_specialization.in_specs_treedef is not None:
      raise ValueError("in_specs already specified")
    in_specs_leaves, in_specs_treedef = tree_util.tree_flatten(in_specs)
    updates["in_specs_treedef"] = in_specs_treedef
    updates["in_specs_leaves"] = tuple(in_specs_leaves)

  if out_specs_fn is not None:
    if base_specialization.out_specs_fn is not None:
      raise ValueError("out_specs_fn already specified")
    updates["out_specs_fn"] = out_specs_fn

  if out_specs is not None:
    if base_specialization.out_specs_treedef is not None:
      raise ValueError("out_specs already specified")
    out_specs_leaves, out_specs_treedef = tree_util.tree_flatten(out_specs)
    updates["out_specs_treedef"] = out_specs_treedef
    updates["out_specs_leaves"] = tuple(out_specs_leaves)

  if devices is not None:
    if base_specialization.devices is not None:
      raise ValueError("devices already specified")
    if isinstance(devices, xc.DeviceList):
      updates["devices"] = devices
    else:
      updates["devices"] = xc.DeviceList(tuple(devices))

  return dataclasses.replace(base_specialization, **updates)


def _get_specs(x: Any) -> ShapeDtypeStructTree:
  """Extracts specs for a pytree of JAX Arrays."""
  shape_dtype_struct = lambda x: api.ShapeDtypeStruct(
      shape=x.shape, dtype=x.dtype, sharding=x.sharding
  )
  return tree_util.tree_map(shape_dtype_struct, x)


def _infer_devices_from_args(args: Any) -> Sequence[jax.Device] | None:
  """Returns a representative device list from a pytree of JAX Arrays."""
  devices = set()
  for x in tree_util.tree_leaves(args):
    devices.add(x.sharding._internal_device_list)
  if not devices:
    return None
  assert len(devices) == 1
  return devices.pop()


def _compile_to_executable(
    name: str,
    fun: Callable[..., Any],
    in_specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    out_specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    devices: xc.DeviceList,
) -> Callable[..., Any]:
  """Compiles a Python function into a runtime executable."""
  # TODO(hyeontaek): Wrap fun as CustomCallProgram and compile it into an
  # executable.
  del name
  del in_specs_leaves
  del out_specs_leaves
  del devices
  return fun


def _make_output_specs_and_push_result_fun(
    info: FunctionInfo, specialization: Specialization, uid: int
) -> Callable[..., Any]:
  """Creates a function that computes output specs and pushes the result to the result store."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.out_specs_treedef is None
  assert specialization.out_specs_leaves is None
  assert specialization.devices is not None

  devices = specialization.devices

  def lowered_fun(*args, **kwargs) -> Sequence[jax.Array]:
    result = info.fun(*args, **kwargs)
    out_specs = _get_specs(result)
    flat_result = tree_util.tree_leaves(result)
    func_backend.SINGLETON_RESULT_STORE.push(uid, flat_result)
    return serialize_specs(out_specs, devices)

  out_specs_leaves, _ = tree_util.tree_flatten(
      make_specs_for_serialized_specs(specialization.devices),
  )
  name = getattr(info.fun, "__name__", "unknown")
  name = f"{name}_output_specs_and_push_result"
  return _compile_to_executable(
      name=name,
      fun=lowered_fun,
      in_specs_leaves=specialization.in_specs_leaves,
      out_specs_leaves=tuple(out_specs_leaves),
      devices=specialization.devices,
  )


def _make_pop_result_fun(
    info: FunctionInfo, specialization: Specialization, uid: int
) -> Callable[..., Any]:
  """Makes a function that pops results from the result store."""
  assert specialization.out_specs_treedef is not None
  assert specialization.out_specs_leaves is not None
  assert specialization.devices is not None

  out_specs_treedef = specialization.out_specs_treedef

  def lowered_fun() -> Any:
    flat_result = func_backend.SINGLETON_RESULT_STORE.pop(uid)
    return tree_util.tree_unflatten(out_specs_treedef, flat_result)

  in_specs_leaves, _ = tree_util.tree_flatten((
      # args
      (),
      # kwargs
      (),
  ))
  name = getattr(info.fun, "__name__", "unknown")
  name = f"{name}_pop_result"
  return _compile_to_executable(
      name=name,
      fun=lowered_fun,
      in_specs_leaves=tuple(in_specs_leaves),
      out_specs_leaves=specialization.out_specs_leaves,
      devices=specialization.devices,
  )


def _make_async_execution_fun(
    info: FunctionInfo, specialization: Specialization
) -> Callable[..., Any]:
  """Makes a function that asynchronously executes the function."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.out_specs_treedef is not None
  assert specialization.out_specs_leaves is not None
  assert specialization.devices is not None

  name = getattr(info.fun, "__name__", "unknown")
  return _compile_to_executable(
      name=name,
      fun=info.fun,
      in_specs_leaves=specialization.in_specs_leaves,
      out_specs_leaves=specialization.out_specs_leaves,
      devices=specialization.devices,
  )


@jax.util.cache(max_size=None)
def _get_specialized_func(
    info: FunctionInfo, specialization: Specialization
) -> Callable[..., Any]:
  """Returns a specialized function for the given specialization."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.devices is not None
  uid = random.getrandbits(63)

  mutex = threading.Lock()
  # Asynchronous execution function that has known output_specs.
  async_execution_func = None

  def specialized_func(*args, **kwargs) -> Any:
    """Specialized function to be executed with given args and kwargs."""
    nonlocal specialization, async_execution_func
    with mutex:
      if async_execution_func is None:
        if specialization.out_specs_treedef is None:
          if specialization.out_specs_fn is None:
            serialized_out_specs = _make_output_specs_and_push_result_fun(
                info, specialization, uid
            )(*args, **kwargs)

            # Waits for the output_specs. This may block.
            out_specs = deserialize_specs(serialized_out_specs)

            # Subsequent calls would use async_execution_func with discovered
            # output_specs.
            specialization = _apply_specialization(
                specialization, out_specs=out_specs
            )
            async_execution_func = _make_async_execution_fun(
                info, specialization
            )

            return _make_pop_result_fun(info, specialization, uid)()
          else:
            # Compute out_specs using out_specs_fn and inputs.
            out_specs = specialization.out_specs_fn(*args, **kwargs)
            specialization = _apply_specialization(
                specialization, out_specs=out_specs
            )
            async_execution_func = _make_async_execution_fun(
                info, specialization
            )
            # Fall-through.
        else:
          async_execution_func = _make_async_execution_fun(info, specialization)
          # Fall-through.

      return async_execution_func(*args, **kwargs)

  return specialized_func


def make_callable(
    fun: Callable[..., Any],
    fun_sourceinfo: str | None,
    fun_signature: inspect.Signature | None,
) -> Callable[..., Any]:
  """Makes a colocated Python callable."""
  return _make_callable(
      FunctionInfo(fun, fun_sourceinfo, fun_signature), Specialization()
  )


def _make_callable(
    info: FunctionInfo,
    specialization: Specialization,
) -> Callable[..., Any]:
  """Internal implementation of make_callable."""

  def specialize(
      in_specs: ShapeDtypeStructTree | None = None,
      out_specs_fn: Callable[..., Any] | None = None,
      out_specs: ShapeDtypeStructTree | None = None,
      devices: Sequence[jax.Device] | None = None,
  ) -> Callable[..., Any]:
    """Returns a colocated Python callable with extra specialization."""
    nonlocal info, specialization
    return _make_callable(
        info,
        _apply_specialization(
            specialization,
            in_specs=in_specs,
            out_specs_fn=out_specs_fn,
            out_specs=out_specs,
            devices=devices,
        ),
    )

  def call(*args, **kwargs) -> Any:
    """Executes the function.

    If the output specs are not known, the very first execution will be
    synchronous.
    """
    nonlocal info, specialization, specialize

    in_specs = _get_specs((args, kwargs))
    if specialization.in_specs_treedef is None:
      if specialization.out_specs_treedef is None:
        # Allow input polymorphism by applying input_specs specialization
        # temporarily for this call.
        return specialize(in_specs=in_specs)(*args, **kwargs)

      # If out_specs is already specialized, we accept only one input_specs
      # permanently by remembering the specialization within this callable
      # itself.
      specialization = _apply_specialization(specialization, in_specs=in_specs)
      # Fall-through.

    if specialization.devices is None:
      devices = _infer_devices_from_args(args)
      if devices is None:
        raise ValueError(
            "No devices found. colocated_python function without input"
            " arguments must be first specialized with devices."
        )
      # Allow device polymorphism by applying devices specialization temporarily
      # for this call.
      return specialize(devices=devices)(*args, **kwargs)

    # If input_specs is known, verify that it matches actual inputs.
    in_specs_leaves, in_specs_treedef = tree_util.tree_flatten(in_specs)
    assert isinstance(specialization.in_specs_treedef, tree_util.PyTreeDef)
    if (
        specialization.in_specs_treedef != in_specs_treedef
        or specialization.in_specs_leaves != tuple(in_specs_leaves)
    ):
      raise ValueError(
          "Input specs do not match: "
          f"Expected ({specialization.in_specs_treedef}, "
          f"{specialization.in_specs_leaves}), "
          f"but got ({in_specs_treedef}, {tuple(in_specs_leaves)})."
      )

    return _get_specialized_func(info, specialization)(*args, **kwargs)

  call = api_boundary(call)
  call = wraps(info.fun)(call)
  call.specialize = specialize
  return call
