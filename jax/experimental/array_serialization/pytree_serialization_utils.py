# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#

# # Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import json
import dataclasses
import collections
import threading
import itertools
import logging
import io
import zipfile
import contextlib
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Callable

import jax
from jax.tree_util import PyTreeDef, default_registry, treedef_is_leaf

PickleModule = ModuleType

logger = logging.getLogger(__name__)

_TREE_REPR_KEY = "__jax_tree_repr"
_LEAF_IDS_KEY = "__jax_leaf_ids"
_LEAF_COUNT_KEY = "__jax_leaf_count"
_NODE_DATA_ARCHIVE_KEY_FORMAT = "___jax_node_data_ref_{}"
_NODE_DATA_ARCHIVE_KEY_REGEX = r"___jax_node_data_ref_([0-9]+)"
_MAX_CONCURRENCY = 32

class _MISSING_TYPE:
  pass
MISSING = _MISSING_TYPE()


def _cls2typerepr(cls):
  return f"{cls.__module__}.{cls.__name__}"

class MemKVStore:
  def __init__(self, data: bytes | None = None):
    self.buffer = io.BytesIO(data) if data is not None else io.BytesIO()
    self.buffer.seek(0)
    self.zipfile = zipfile.ZipFile(
      self.buffer, mode="a", compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    self._lock, self._closed = threading.Lock(), False

  def keys(self) -> list[str]:
    assert not self._closed
    return self.zipfile.namelist()

  def tobytes(self) -> bytes:
    assert not self._closed
    with self._lock:
      self.zipfile.close()
      self._closed = True
      return self.buffer.getvalue()

  def write(self, filename: str | os.PathLike[str], data: bytes | str) -> None:
    assert not self._closed
    with self._lock:
      self.zipfile.writestr(str(filename), data)

  def read(self, filename: str | os.PathLike[str]) -> bytes:
    assert not self._closed
    return self.zipfile.read(str(filename))

identity = lambda x: x

class SerializationRegistry:
  def __init__(self):
    self._in_place = {}
    self._serialization_map, self._deserialization_map = {}, {}
    self._class_map = {}

    # in-built leaf types
    for t in [int, float, str, complex, bool]:
      self._serialization_map[_cls2typerepr(t)] = (json.dumps, "json")
      self._deserialization_map["json"] = json.loads
      self._class_map[_cls2typerepr(t)] = t
    for t in [bytes, bytearray]:
      self._serialization_map[_cls2typerepr(t)] = (identity, "bin")
      self._deserialization_map["bin"] = identity
      self._class_map[_cls2typerepr(t)] = t

    # in-built nodes
    for t in [tuple, list, dict, collections.OrderedDict]:
      type_repr = _cls2typerepr(t)
      self._in_place[type_repr] = True
      self._serialization_map[type_repr] = (identity, type_repr)
      self._deserialization_map[type_repr] = lambda data: (
        tuple(data) if isinstance(data, list) else data)  # dicts need key tuple
      self._class_map[type_repr] = t

    # fallback functions
    self._fallback_serialize_fn: Callable[[str, Any], str | bytes] | None = None
    self._fallback_deserialize_fn: Callable[[str, str | bytes], Any] | None = (
      None)

  def register_type(self, cls: type[Any],
                    serialize_fn: Callable[[Any], Any] | None = None,
                    deserialize_fn: Callable[[Any], Any] | None = None,
                    name: str | None = None, in_place: bool = False):
    name = name if name is not None else _cls2typerepr(cls)
    self._serialization_map[name] = (serialize_fn, name)
    self._deserialization_map[name] = deserialize_fn
    self._in_place[name] = in_place
    self._class_map[name] = cls

  def deregister_type(self, cls: type[Any], name: str | None = None):
    name = name if name is not None else _cls2typerepr(cls)
    del self._serialization_map[name]
    del self._deserialization_map[name]
    del self._in_place[name]
    del self._class_map[name]

  def serialize(self, obj: Any, name: str | None = None
                ) -> tuple[bytes | str, str]:
    cls = type(obj)
    name = name if name is not None else _cls2typerepr(cls)
    if name in self._serialization_map:
      serialization_fn, method_name = self._serialization_map[name]
      return (serialization_fn(obj), method_name)
    elif self._fallback_serialize_fn is not None:
      return self._fallback_serialize_fn(name, obj), "fallback"
    else:
      raise ValueError(f"Name \"{name}\" not registered for serialization"
                       f" (data type = {cls}).")

  def deserialize(self, data: bytes, method_name: str) -> Any:
    if method_name in self._deserialization_map:
      return self._deserialization_map[method_name](data)
    elif self._fallback_deserialize_fn is not None:
      return self._fallback_deserialize_fn(method_name, data)
    else:
      raise ValueError(f"Extension `{method_name}` not registered for"
                      " deserialization.")

  def get_type(self, name: str) -> type[Any] | _MISSING_TYPE:
    return self._class_map.get(name, MISSING)

  def register_fallback(self, serialize_fn: Callable[[Any], str | bytes],
                        deserialize_fn: Callable[[str | bytes], Any]):
    def _fallback_serialize_fn(class_name: str, obj: Any) -> str | bytes:
      del class_name
      return serialize_fn(obj)

    def _fallback_deserialize_fn(class_name: str, data: str | bytes) -> Any:
      del class_name
      return deserialize_fn(data)

    self._fallback_serialize_fn = _fallback_serialize_fn
    self._fallback_deserialize_fn = _fallback_deserialize_fn

  def deregister_fallback(self):
    self._fallback_serialize_fn = None
    self._fallback_deserialize_fn = None

  def copy(self):
    new_registry = SerializationRegistry()
    new_registry._serialization_map = self._serialization_map.copy()
    new_registry._deserialization_map = self._deserialization_map.copy()
    new_registry._fallback_serialize_fn = self._fallback_serialize_fn
    new_registry._fallback_deserialize_fn = self._fallback_deserialize_fn
    new_registry._in_place = self._in_place.copy()
    new_registry._class_map = self._class_map.copy()
    return new_registry

  @contextlib.contextmanager
  def with_fallback(self, dumps: Callable[[Any], str | bytes],
                    loads: Callable[[str | bytes], Any]):
    try:
      self.register_fallback(dumps, loads)
      yield
    finally:
      self.deregister_fallback()

  def in_place(self, name: str) -> bool:
    return self._in_place.get(name, False)

default_serialization_registry = SerializationRegistry()

def _node_serialize(node_data_store: dict[str, Any], counter: itertools.count,
                    node_type_data: tuple[type[Any], Any],
                    registry: SerializationRegistry,
                    executor: ThreadPoolExecutor) -> tuple[str, Any]:
  node_type, node_data = node_type_data
  type_repr = _cls2typerepr(node_type)
  if registry.in_place(type_repr):
    return (type_repr, node_data)
  else:
    id = next(counter)
    name = _NODE_DATA_ARCHIVE_KEY_FORMAT.format(id)
    assert node_data_store is not None, (
        "Archive must be provided for not in-tree node data.")

    node_data_store[f"{name}.{type_repr}"] = executor.submit(
        lambda: registry.serialize(node_data, name=type_repr)[0])
    return (type_repr, name)

def _node_deserialize(node_data_store: dict[str, Any],
                      node_type_data: tuple[str, Any],
                      registry: SerializationRegistry, best_effort: bool = False
                      ) -> tuple[type[Any], Any]:
  type_repr, node_data = node_type_data
  try:
    node_type = registry.get_type(type_repr)
    if node_type is MISSING:
      raise ValueError(f"Node type \"{type_repr}\" is not registered. Register"
                       "it via"
                       " `default_serialization_register.register_type(...).")

    # if the node has an in-tree representation, just return that data
    if registry.in_place(type_repr):
      node_data = registry.deserialize(node_data, type_repr)
      return (node_type, node_data)  # type: ignore

    node_id = re.match(_NODE_DATA_ARCHIVE_KEY_REGEX, node_data)
    if node_id is None:
      raise ValueError(f"Node type {type_repr} is not registered as in-tree,"
                        f" but the store key {node_data} does not match the"
                        f" expected format. Check that the node was registered"
                        f" that same way as during serialization. Otherwise use"
                        f" use argument `best_effort=True` to reconstruct as"
                        f" a list of children.")
    node_id = int(node_id.group(1))  # type: ignore
    filename = f"{_NODE_DATA_ARCHIVE_KEY_FORMAT.format(node_id)}.{type_repr}"
    payload = node_data_store[filename]
    node_data = registry.deserialize(payload, type_repr)
    return (node_type, node_data)  # type: ignore
  except Exception as e:  # pylint: disable=broad-except
    if best_effort:
      logger.warning("We couldn't read the node %s, returning list of children",
                     type_repr)
      return (list, None)
    else:
      raise e

################################################################################

def _serialize_pytree_helper(node, leaf_counter: itertools.count,
                             node_counter: itertools.count,
                             node_data_store: dict[str, Any],
                             registry: SerializationRegistry,
                             executor: ThreadPoolExecutor):
  if treedef_is_leaf(node) and node.num_leaves == 1:
    return dict(node_type="leaf", leaf_id=next(leaf_counter))
  node_repr = dict()
  type_repr, node_data = _node_serialize(node_data_store, node_counter,
                                         node.node_data(), registry, executor)
  node_repr["name"], node_repr["node_data_ref"] = type_repr, node_data
  node_repr["node_type"] = "static_node" if node.num_nodes == 1 else "node"
  node_repr["children"] = [_serialize_pytree_helper(  # type: ignore
      child, leaf_counter, node_counter, node_data_store, registry, executor)
                           for child in node.children()]
  return node_repr

@jax.tree_util.register_static
class _EmptyStaticNode:
  pass

def _deserialize_pytree_helper(node, node_data_store: dict[str, Any],
                               registry: SerializationRegistry,
                               best_effort: bool = False):
  assert "node_type" in node

  if node["node_type"] == "leaf":
    # case 1: normal leaf node -------------------------------------------------
    node_data, pytree_children = None, ()
  else:
    node_data = _node_deserialize(node_data_store,
                                  (node["name"], node["node_data_ref"]),
                                  registry, best_effort=best_effort)

    pytree_children = [_deserialize_pytree_helper(
      child, node_data_store, registry, best_effort=best_effort)
      for child in node["children"]] # type: ignore
    if (node["node_type"] == "static_node" and best_effort
        and node_data[0] is list):
      # the node failed to deserialize and was replaced at best effort with list
      return jax.tree.structure(_EmptyStaticNode())
  pt = PyTreeDef.make_from_node_data_and_children(default_registry, node_data,
                                                  pytree_children)
  return pt


# serialize and deserialize pytree methods namespaces: permissive and strict
def serialize_pytree(node, registry: SerializationRegistry | None = None
                     ) -> tuple[dict[str, Any], int, dict[str, Any]]:
  node_data_store: dict[str, Any] = {}
  executor = ThreadPoolExecutor(max_workers=_MAX_CONCURRENCY)
  registry = (registry if registry is not None
              else default_serialization_registry.copy())
  leaf_counter, node_counter = itertools.count(), itertools.count()
  root_repr = _serialize_pytree_helper(
      node, leaf_counter, node_counter, node_data_store, registry, executor)
  leaf_count = next(leaf_counter)
  tree_repr = {_TREE_REPR_KEY: root_repr, _LEAF_COUNT_KEY: leaf_count,
                _LEAF_IDS_KEY: list(range(leaf_count))}

  # gather data from the thread pool executor
  node_data_store = {k: v.result() for k, v in node_data_store.items()}
  return tree_repr, leaf_count, node_data_store

def deserialize_pytree(rawtree: dict[str, Any],
                       node_data_store: dict[str, Any] | None = None,
                       registry: SerializationRegistry | None = None,
                       best_effort: bool = False) -> Any:
  node_data_store = {} if node_data_store is None else node_data_store
  registry = (registry if registry is not None
              else default_serialization_registry.copy())
  pt = _deserialize_pytree_helper(rawtree[_TREE_REPR_KEY], node_data_store,
                                  registry, best_effort=best_effort)
  leaf_ids = rawtree[_LEAF_IDS_KEY]
  return jax.tree.unflatten(pt, leaf_ids)
