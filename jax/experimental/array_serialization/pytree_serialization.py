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

from os import PathLike
import os
from types import ModuleType
import re
from typing import Any
from uuid import uuid4, UUID
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import shutil
import logging

import jax
from jax.tree_util import PyTreeDef
from jax.util import safe_zip
from jax._src import distributed
from jax._src.layout import Layout
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import asyncio_utils
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
import jax.experimental.array_serialization.pytree_serialization_utils as utils
from jax.sharding import SingleDeviceSharding
from jax._src.path import epath_installed, Path
import numpy as np

_ORDERED_THREAD_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_THREADING_SAVE_LOCK = threading.Lock()

_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_PYTREEDEF_FILE = "pytreedef.json"
_TENSORSTORE_SUFFIX = ".tensorstore"
_LEAF_DATA_DIR = "leaf_data"
_OBJ_DATA_ARCHIVE = "obj_data.zip"
_NODE_DATA_ARCHIVE = "node_data.zip"
_TYPE_ID_LEAF_DELIMITER = " -> "
_USE_OCDBT = True  # a lot of the code relies on this being True
_MAX_PATH_LENGTH = 4096
_ARRAY_STORE_DIRNAME = f"array_store{_TENSORSTORE_SUFFIX}"
_ARRAY_TYPE_NAME = "Array"
_ARRAY_TYPE_REGEX = r"Array\[\[([0-9, ]*)\],\s*([a-zA-Z0-9_]+)\]"
_DOT_REPLACEMENT = ":"

__all__ = ["save", "load", "load_pytree",
           "nonblocking_load", "nonblocking_save"]

PyTreeT = Any
PickleModule = ModuleType

logger = logging.getLogger(__name__)

def _get_sync_client() -> distributed.xla_extension.DistributedRuntimeClient:
  assert jax.process_count() > 1, (
    "You are attempting to wait for other hosts, but there is only 1 host"
  )
  assert distributed.global_state.client is not None, (
    "The distributed runtime is not initialized. You likely need to call"
    " `jax.distributed.initialize()` first."
  )
  return distributed.global_state.client

def _get_unique_sync_key() -> str | None:
  """Generate a thread-local key for ensuring all host finish (de)serializing"""
  if jax.process_count() == 1:
    return None
  # broadcast a thread-local unique barrier name
  sync_key_id = UUID(bytes=np.array(multihost_utils.broadcast_one_to_all(
    np.frombuffer(uuid4().bytes, dtype=np.int32))).tobytes())
  sync_key = f"jax_sync_key_{str(sync_key_id)}"
  return sync_key

def _is_str_same_on_all_hosts(path: str | PathLike[str]) -> bool:
  """All-gather the location of the checkpoint and check if it's the same."""
  if jax.process_count() <= 1:
    return False
  path_b = str(path).encode("utf-8")
  assert len(path_b) <= _MAX_PATH_LENGTH, (
      f"Path exceeds maximum length of {_MAX_PATH_LENGTH} in multiprocess"
      " case.")
  path_array = np.concatenate([
      np.frombuffer(path_b, dtype=np.uint8), np.zeros(
          _MAX_PATH_LENGTH - len(path_b), dtype=np.uint8)])
  all_path_arrays = multihost_utils.process_allgather(path_array)
  return bool(np.all(all_path_arrays == all_path_arrays[:1, ...]))

def _sync_on_key(key: str | None, extra_tag: str = "") -> None:
  if key is None:
    return
  full_key = key if not extra_tag else f"{key}-{extra_tag}"
  multihost_utils.sync_global_devices(full_key)

def _is_array_like(x):
  return isinstance(x, (jax.Array, np.ndarray))

def _leaf_to_type_desc(leaf) -> str:
  if leaf is None:
    return "null"
  elif isinstance(leaf, (np.ndarray, jax.Array)):
    return (f"{_ARRAY_TYPE_NAME}[[{', '.join(map(str, leaf.shape))}]," +
            f" {leaf.dtype.name}]")
  else:
    return type(leaf).__name__

def _leaf_desc_to_leaf(leaf_desc: str) -> str | jax.ShapeDtypeStruct:
  leaf_type: str = (leaf_desc.split(_TYPE_ID_LEAF_DELIMITER, 1)[0]
                    if _TYPE_ID_LEAF_DELIMITER in leaf_desc else leaf_desc)
  if not leaf_type.startswith(_ARRAY_TYPE_NAME):
    return leaf_type
  shape_dtype_match = re.match(_ARRAY_TYPE_REGEX, leaf_type)
  assert shape_dtype_match is not None, (
      f"Failed to parse array descriptor: {leaf_type} with pattern:"
      f" {_ARRAY_TYPE_REGEX}")
  shape_str, dtype_str = shape_dtype_match.groups()
  shape = [int(x.strip()) for x in shape_str.strip("]").strip().split(",")
            if len(x.strip()) > 0]
  dtype = jax.numpy.dtype(dtype_str)
  return jax.ShapeDtypeStruct(shape, dtype)

def _join_leaf_type_and_id(leaf_type: str, leaf_id: str) -> str:
  return f"{leaf_type}{_TYPE_ID_LEAF_DELIMITER}{leaf_id}"

def _inscribe_leaf_types(pytree_repr: dict[str, Any],
                         leaf_id_type_map: dict[str, str]):
  """Rewrite a JSON PyTree representation by adding type to leaf_id."""
  if pytree_repr["node_type"] == "leaf":
    leaf_id = pytree_repr["leaf_id"]
    if leaf_id is None:
      return
    pytree_repr["leaf_id"] = _join_leaf_type_and_id(leaf_id_type_map[leaf_id],
                                                    leaf_id)
  else:
    _ = [_inscribe_leaf_types(child, leaf_id_type_map)
         for child in pytree_repr["children"]]

def _inplace_add_types_to_pytree_repr(pytree_repr, leaf_ids_flat, data_flat):
  # inscribe types into leaf ids in-place
  leaf_ids_type_map = {leaf_id: _leaf_to_type_desc(leaf) for (leaf_id, leaf)
                       in safe_zip(leaf_ids_flat, data_flat)}
  _inscribe_leaf_types(pytree_repr[utils._TREE_REPR_KEY], leaf_ids_type_map)
  pytree_repr[utils._LEAF_IDS_KEY] = [
    _join_leaf_type_and_id(leaf_ids_type_map[leaf_id], leaf_id)
    for leaf_id in leaf_ids_flat]

def _combine_two_pytrees(dest: dict[str, Any], source: dict[str, Any]):
  """Combine two pytrees in JSON format, in-place into dest."""
  assert dest["node_type"] == source["node_type"]
  if dest["node_type"] == "leaf":
    _, id1 = dest["leaf_id"].split(_TYPE_ID_LEAF_DELIMITER, 1)
    type2, id2 = source["leaf_id"].split(_TYPE_ID_LEAF_DELIMITER, 1)
    assert id1 == id2
    if type2 != "null":
      dest["leaf_id"] = source["leaf_id"]
    return dest
  else:
    dest["children"] = [
      _combine_two_pytrees(child1, child2) for (child1, child2)
      in safe_zip(dest["children"], source["children"])]
    return dest


def _is_remote_path(path: str | PathLike[str]):
  # we check whether a path is remote by checking the prefix
  # we need to truncate e.g., gs:// to gs:/ because pathlib.Path collapses //
  return any(str(path).startswith(prefix[:-1])
             for prefix in _REMOTE_URL_PREFIXES)

def _rm_dir(root: Path) -> None:
  if _is_remote_path(root):
    root.rmtree()  # pytype: disable=attribute-error
  else:
    shutil.rmtree(root)

def _maybe_overwrite_or_error(root: Path, overwrite: bool, partial_write: bool,
                              pytree_repr: dict[str, Any],
                              distinct_locations: bool, sync_key: str | None
                              ) -> dict[str, Any]:
  if overwrite:
    if root.exists() and len(list(root.iterdir())) > 0:
      # check that we're only deleting things that come from JAX
      # refuse to rm directories containing additional entries
      paths_present = list(root.iterdir())
      extra_member_paths = [path for path in paths_present if path.name not in
                            (_PYTREEDEF_FILE, _LEAF_DATA_DIR,
                             _NODE_DATA_ARCHIVE)]

      assert len(extra_member_paths) == 0, (
        "Refusing to work on a directory that is not a previous checkpoint."
        f" Unrecognized paths: {extra_member_paths}. Remove them manually if"
        f" you're sure you want to use {root} as the checkpoint directory.")

      if partial_write and Path(_PYTREEDEF_FILE) in [path.relative_to(root)
                                                     for path in paths_present]:
        try:
          other_pytree = json.loads((root / _PYTREEDEF_FILE).read_text())
          assert other_pytree[utils._LEAF_COUNT_KEY] == pytree_repr[
              utils._LEAF_COUNT_KEY]
          tree_repr = _combine_two_pytrees(other_pytree[utils._TREE_REPR_KEY],
                                           pytree_repr[utils._TREE_REPR_KEY])
          new_leaf_ids = [y if not y.startswith("null") else x
                          for x, y in safe_zip(
                              other_pytree[utils._LEAF_IDS_KEY],
                              pytree_repr[utils._LEAF_IDS_KEY])]
          return {utils._TREE_REPR_KEY: tree_repr,
                  utils._LEAF_COUNT_KEY: pytree_repr[utils._LEAF_COUNT_KEY],
                  utils._LEAF_IDS_KEY: new_leaf_ids}
        except AssertionError:
          logger.warning("The previous pytree does not match, overwritting"
                         " existing data.")
      if (jax.process_index() == 0 or distinct_locations) and root.exists():
        _rm_dir(root)
    _sync_on_key(sync_key, "overwrite")
    return pytree_repr
  else:
    if (root.exists() and len(list(root.iterdir())) > 0):  # not empty
      raise NotImplementedError(f"Files already exist at path: `{root}`,"
                                f" but you specified `{overwrite=}`")
    return pytree_repr


def _obj_serialize(archive: utils.MemKVStore, filename_id: str | int, x: Any,
                   registry: utils.SerializationRegistry) -> None:
  """Serialization method for NOT-array objects."""
  # we're only interested in name and suffix
  filename = Path(Path(str(filename_id)).name)
  serialization_registry = registry
  if _is_array_like(x):
    raise ValueError(
        "Arrays cannot be serialized using this method for non-arrays.")
  # dispatch the serialization method in a thread to yield async control
  payload, method_name = serialization_registry.serialize(x)
  suffix = "." + method_name.lstrip(".").replace(".", _DOT_REPLACEMENT)
  archive.write(filename.with_suffix(suffix), payload)

def _obj_deserialize(archive: utils.MemKVStore, filename: str,
                     registry: utils.SerializationRegistry,
                     best_effort: bool = False) -> Any:
  """Deserialization method for NON-array objects."""
  path = Path(filename)
  payload = archive.read(path)
  method_name = str(path.suffix).lstrip(".").replace(_DOT_REPLACEMENT, ".")
  try:
    return registry.deserialize(payload, method_name)
  except ValueError as exc:
    if best_effort:
      logging.warning("Unrecognized data type `%s` we'll do our best and just"
                      " return the raw bytes", method_name)
      return payload
    else:
      raise exc

async def serialize_array(arr, path, extra_config, distinct_locations: bool) -> None:
  arr = jax.numpy.asarray(arr, dtype=arr.dtype)
  extra_ts_spec = extra_config
  process_num = (jax.process_index() if (
      jax.process_count() > 1 and not distinct_locations) else None)
  default_ts_spec = ts_impl.get_tensorstore_spec(
      path, ocdbt=_USE_OCDBT, process_num=process_num, arr=arr)
  expected_path = default_ts_spec['kvstore']['base']['path']
  ts_spec = ts_impl.merge_nested_specs(default_ts_spec, extra_ts_spec)
  ts_impl.verify_tensorstore_spec(ts_spec, arr, expected_path,
                                  check_metadata=True)
  # all hosts write because they're writing to different storage locations (to
  # be combined later) -> `primary_host=None`
  await ts_impl.async_serialize(arr, ts_spec, primary_host=None)

def finalize_array_store(kvstore_path, extra_config, distinct_locations: bool
                         ) -> None:
  # only in multiprocess case and only process 0
  if distinct_locations or jax.process_count() <= 1 or jax.process_index() != 0:
    return
  extra_ts_spec = extra_config
  dummy_key_path = os.path.join(kvstore_path, "dummy_key")
  combined_ts_spec = ts_impl.merge_nested_specs(ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=_USE_OCDBT, process_num=None), extra_ts_spec)
  children_ts_spec = [ts_impl.merge_nested_specs(ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=_USE_OCDBT, process_num=i), extra_ts_spec)
      for i in range(jax.process_count())]
  combined_kvstore = combined_ts_spec["kvstore"]
  children_kvstores = [ts_spec["kvstore"] for ts_spec in children_ts_spec]
  _ = combined_kvstore.pop("path")
  _ = [kvstore.pop("path") for kvstore in children_kvstores]
  asyncio.run(ts_impl.combine_kvstores(combined_kvstore, children_kvstores))

async def deserialize_array(
    path: str | PathLike[str], sharding: jax.sharding.Sharding | Layout,
    ts_spec: dict[str, Any],
    byte_limiter: asyncio_utils._LimitInFlightBytes | None = None) -> jax.Array:
  # every process reads from the central location
  default_ts_spec = ts_impl.get_tensorstore_spec(
      path, ocdbt=_USE_OCDBT, process_num=None)
  expected_path = default_ts_spec['kvstore']['base']['path']
  ts_spec = ts_impl.merge_nested_specs(default_ts_spec, ts_spec)
  ts_impl.verify_tensorstore_spec(ts_spec, arr=None, path=expected_path,
                                  check_metadata=False)
  return await ts_impl.async_deserialize(sharding, ts_spec,
                                         byte_limiter=byte_limiter)

def save(data: PyTreeT, directory: str | PathLike[str], overwrite: bool = True,
         partial_write: bool = False, ts_specs: PyTreeT | None = None) -> None:
  """Saves the given data structure to the provided directory path.

  This function provides functionality to serialize and save a data structure
  comprising JAX arrays, NumPy arrays, Python objects, etc., along with its
  structure to a given directory. It leverages `PyTree` for flattening and
  reconstructing the data structure.

  Args:
    data: The data structure to be saved. Arbitrary composition of JAX arrays,
      NumPy arrays, and Python objects, including nested structures.
    directory: The directory path where the data will be saved. A local path or
      a remote URL (e.g., gs://, s3://). For remote URLs, `etils` is required.
    overwrite: If True, any existing directory with the same name will be
      overwritten.
  Raises:
    AssertionError: If attempting to save to a remote path without the `etils`
      package installed.
    NotImplementedError: If `overwrite` is False and a checkpoint already
      exists at the provided directory.
  """
  with _THREADING_SAVE_LOCK:
    return _save(data, directory, overwrite, partial_write, ts_specs)

def _save(data: PyTreeT, directory: str | PathLike[str], overwrite: bool = True,
         partial_write: bool = False, ts_specs: PyTreeT | None = None) -> None:
  sync_key = _get_unique_sync_key()  # get a synchronization key for multi-host

  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`."
  )
  data_flat, pytreedef = jax.tree.flatten(data, is_leaf=lambda x: x is None)
  distinct_locations = not _is_str_same_on_all_hosts(directory)
  if jax.process_count() > 1 and distinct_locations:
    logger.warning("Saving to different locations on different hosts is"
                   " supported, but extremely fragile. Consider using a single"
                   " location.")

  # start serialization ##################################
  futures, executor = [], ThreadPoolExecutor(max_workers=utils._MAX_CONCURRENCY)
  serialization_registry = utils.default_serialization_registry.copy()

  # 0. serialize the pytree
  pytree_repr, leaf_count, node_data_store = utils.serialize_pytree(
    pytreedef, registry=serialization_registry)
  leaf_ids_flat = list(range(leaf_count))
  _inplace_add_types_to_pytree_repr(pytree_repr, leaf_ids_flat, data_flat)

  # overwrite or error
  root = Path(directory).resolve()
  pytree_repr = _maybe_overwrite_or_error(
    root, overwrite, partial_write, pytree_repr, distinct_locations, sync_key)

  if not _is_remote_path(directory):
    if distinct_locations or jax.process_index() == 0:
      root.mkdir(exist_ok=True)  # do not make parents, that's too much
      assert root.exists() and root.is_dir()
  _sync_on_key(sync_key, "mkdir")

  def _write_pytree():
    if not (jax.process_index() == 0 or distinct_locations):
      return
    # augment the pytree representation with leaf types
    (root / _PYTREEDEF_FILE).write_text(json.dumps(pytree_repr, indent=2))

  futures.append(executor.submit(_write_pytree))

  # 1. serialize non-array (objects) in the pytree
  def _write_objects():
    if not (jax.process_index() == 0 or distinct_locations):
      return
    archive_path = root / _LEAF_DATA_DIR / _OBJ_DATA_ARCHIVE
    archive_data = None
    if overwrite and archive_path.is_file() and archive_path.exists():
      archive_data = archive_path.read_bytes()
    obj_archive = utils.MemKVStore(archive_data)
    objs_and_paths = [
        (data, leaf_id) for leaf_id, data in safe_zip(leaf_ids_flat, data_flat)
        if not _is_array_like(data) and data is not None]
    _ = list(executor.map(lambda arg: _obj_serialize(*arg),
                     [(obj_archive, leaf_id, data, serialization_registry)
                      for data, leaf_id in objs_and_paths]))
    archive_path.parent.mkdir(exist_ok=True)
    archive_path.write_bytes(obj_archive.tobytes())  # always write

  futures.append(executor.submit(_write_objects))

  # 2. serialize arrays
  arrs_and_paths = [(data, root / _LEAF_DATA_DIR / _ARRAY_STORE_DIRNAME /
                     str(leaf_id)) for leaf_id, data in safe_zip(
                         leaf_ids_flat, data_flat) if _is_array_like(data)]
  ts_specs = (([None] * len(arrs_and_paths)) if ts_specs is None else
              jax.tree.leaves(ts_specs,
                              is_leaf=ts_impl.is_tensorstore_spec_leaf))

  async def _serialize_arrays():
    await asyncio.gather(*[serialize_array(
      arr, path, extra_ts_spec, distinct_locations)
      for ((arr, path), extra_ts_spec) in safe_zip(arrs_and_paths, ts_specs)])

  futures.append(executor.submit(asyncio.run, _serialize_arrays()))

  # 3. serialize node data if permissive #################
  def _write_node_data():
    if not (jax.process_index() == 0 or distinct_locations):
      return
    node_data_archive = utils.MemKVStore()
    for node_data_key, node_data_bytes in node_data_store.items():
      # this KV storage only admits sequential writes
      node_data_archive.write(node_data_key, node_data_bytes)
    # wait on all node_data serialize futures to finish before writing to disk
    archive_path = root / _NODE_DATA_ARCHIVE  # always write
    archive_path.write_bytes(node_data_archive.tobytes())

  futures.append(executor.submit(_write_node_data))

  _ = [fut.result() for fut in futures]
  _sync_on_key(sync_key, "serialization")
  if len(arrs_and_paths) > 0:
    store_path = arrs_and_paths[0][1].parent
    finalize_array_store(store_path, ts_specs[0], distinct_locations)
  # we are done with all async ops here, we can block
  _sync_on_key(sync_key, "end")

def load(directory: str | PathLike[str],
         shardings: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
         pytree: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
         ts_specs: PyTreeT | None = None, best_effort: bool = False) -> PyTreeT:
  """Loads and reconstructs a data structure from a directory.

  Args:
    directory: Directory path where the data is stored.
    shardings: Sharding strategy for array objects. If None, defaults to
      single device sharding on the default device.
    pytree: Optional pre-populated PyTree for structure. If provided, must
      specify a pytree with string object ids. Useful for partial reads.
    best_effort: Proceed with deserialization even in the face of partial
      failures. Return custom nodes as a list of children.
  Returns:
    Reconstructed data structure.
  Raises:
    AssertionError: If attempting to load from a remote path without etils
      installed.
    ValueError: If data for specific leaf IDs is missing in the directory.
    ImportError: If supported node type (e.g., flax's FrozenDict) cannot be
      imported.
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")

  root = Path(directory)
  assert root.is_dir(), f"Checkpoint directory {root} does not exist"
  if not _is_remote_path(root):
    root = root.resolve()

  # deserialize in 3 stages

  # 1. deserialize PyTree (if permissive inserting node_data)
  if pytree is utils.MISSING:
    pytreedef = load_pytree(directory, best_effort)
    # in pytreedef, None leafs indicate StaticNodes (without leaves)
    # so we CANNOT flatten with is_leaf=lambda x: x is None
    leaf_ids, pytreedef = jax.tree.flatten(pytreedef)
  else:
    leaf_ids, pytreedef = jax.tree.flatten(pytree, is_leaf=lambda x: x is None)
  obj_leaf_ids = [int(leaf_id.split(_TYPE_ID_LEAF_DELIMITER, 1)[1]) for leaf_id
                  in leaf_ids if leaf_id is not None
                  and not leaf_id.startswith(_ARRAY_TYPE_NAME)
                  and not leaf_id.startswith("null")]
  serialization_registry = utils.default_serialization_registry.copy()
  executor = ThreadPoolExecutor(max_workers=utils._MAX_CONCURRENCY)

  # 2. deserialize non-array objects
  def _deserialize_objs():
    obj_archive = utils.MemKVStore(data=(
      root / _LEAF_DATA_DIR / _OBJ_DATA_ARCHIVE).read_bytes())
    _key2id = lambda x: int(Path(x).stem)
    obj_keys = list(obj_archive.keys())
    missing_leaf_ids = set(obj_leaf_ids) - set(map(_key2id, obj_keys))
    requested_obj_keys = [obj_key for obj_key in obj_keys
                          if _key2id(obj_key) in obj_leaf_ids]
    if len(missing_leaf_ids) > 0:
      raise ValueError(
        f"Values {missing_leaf_ids} are missing from the checkpoint directory.")
    obj_futs = [executor.submit(_obj_deserialize, obj_archive, obj_key,
                                registry=serialization_registry,
                                best_effort=best_effort)
                for obj_key in requested_obj_keys]
    obj_values = [fut.result() for fut in obj_futs]
    return dict(safe_zip(map(_key2id, requested_obj_keys), obj_values))

  objs_fut = executor.submit(_deserialize_objs)

  # 3. deserialize array objects
  arr_leaf_ids = [leaf_id.split(_TYPE_ID_LEAF_DELIMITER, 1)[1] for leaf_id
                  in leaf_ids if leaf_id is not None
                  and leaf_id.startswith(_ARRAY_TYPE_NAME)]
  arr_paths = [root / _LEAF_DATA_DIR / _ARRAY_STORE_DIRNAME / leaf_id
               for leaf_id in arr_leaf_ids]
  # missing sharding assumes we want to deserialize on default device
  if shardings is utils.MISSING:
    device = jax.devices()[0]  # default device
    shardings = [SingleDeviceSharding(device) for _ in arr_paths]
  else:
    shardings = jax.tree.flatten(shardings)[0]
    assert len(shardings) == len(arr_paths), (
      "The sharding leaves must match the load arrays requested.")
  ts_specs = (([None] * len(arr_paths)) if ts_specs is None else
              jax.tree.leaves(ts_specs,
                              is_leaf=ts_impl.is_tensorstore_spec_leaf))
  byte_limiter = asyncio_utils._LimitInFlightBytes(100 * 1024 ** 3)  # 100 GB
  async def _deserialize_arrays():
    return await asyncio.gather(*[
        deserialize_array(path, sharding, ts_spec, byte_limiter)
        for (path, sharding, ts_spec)
        in safe_zip(arr_paths, shardings, ts_specs)])

  arr_keys = [int(path.stem) for path in arr_paths]
  arr_values_fut = executor.submit(asyncio.run, _deserialize_arrays())

  # finally, collect the results
  arrs = dict(zip(arr_keys, arr_values_fut.result()))
  objs = objs_fut.result()
  arr_and_objs = arrs | objs
  filled_values = [arr_and_objs.get(leaf_id, None)
                   for leaf_id in range(len(leaf_ids))]
  return jax.tree.unflatten(pytreedef, filled_values)

def load_pytree(directory: str | PathLike[str], best_effort: bool = False
                ) -> PyTreeDef:
  """Loads a pytree from the given directory.
  Args:
    directory: Directory path to load from.
    best_effort: Proceed with deserialization even in the face of partial
      failures. Return custom nodes as a list of children.
  Returns:
    The loaded pytree.
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  root = Path(directory)
  json_content = (root / _PYTREEDEF_FILE).read_text()
  raw_tree = json.loads(json_content)
  serialization_registry = utils.default_serialization_registry.copy()
  executor = ThreadPoolExecutor(max_workers=utils._MAX_CONCURRENCY)
  if (root / _NODE_DATA_ARCHIVE).exists():
    node_data_archive = utils.MemKVStore(data=(root / _NODE_DATA_ARCHIVE
                                               ).read_bytes())
    node_data_futs = {k: executor.submit(node_data_archive.read, k)
                      for k in node_data_archive.keys()}
    node_data_store = {k: v.result() for k, v in node_data_futs.items()}
  else:
    node_data_store = dict()
  return utils.deserialize_pytree(raw_tree, node_data_store,
                                  registry=serialization_registry,
                                  best_effort=best_effort)


class SerializationFuture:
  """Keeps track of saving/loading serialized data via:
    - self.done() - non-blocking check whether the underlying coroutine finished
    - self.result() - gets the result of the coroutine, raises error if not done
    - self.pytree - the property describing the data overview (short leaf desc.)

  The class takes in an async_fn and args/kwargs for it and launches it
  immediately in a separate Python thread. This allows it to work both in sync
  as well as in an async environment.
  """
  def __init__(self, fn, *args, **kw):
    self._pytree = None
    # create a closure which will run an asyncio routine in separate thread
    # and will populate either self._retval if no errors were raised or
    # self._exception if there were errors

    # running an asyncio routine in a thread is a reliable way of scheduling
    # an asyncio routine in the background both in regular synchronous contexts
    # but also in (unexpectedly?) asynchronous contexts like Jupyter Notebooks
    self._retval, self._exception = None, None
    self._done_event = threading.Event()
    self._fn = fn

    def _run_in_thread():
      ret, exc = None, None
      try:
        ret = self._fn(*args, **kw)
      except Exception as e:  # pylint: disable=broad-except
        exc = e
      self._done_event.set()
      # populate either the result or the exception
      self._retval, self._exception = ret, exc

    self._thread = threading.Thread(target=_run_in_thread)
    self._thread.start()
    # do not join the thread

  @property
  def pytree(self):
    return self._pytree

  @pytree.setter
  def pytree(self, new_pytree: PyTreeDef):
    msg = f"You cannot set the .pytree property in {type(self)} more than once."
    assert self._pytree is None or self._pytree == new_pytree, msg
    self._pytree = new_pytree

  def done(self):
    """Check if the underlying deserialization is done. Non-blocking."""
    # return not self._thread.is_alive()
    return self._done_event.is_set()

  def result(self):
    """Retrieve the result or raise an exception of the `async_fn`."""
    assert self.done()
    self._thread.join()
    if self._exception is not None:  # exception has been raised
      raise self._exception
    return self._retval

  def __await__(self):
    while not self.done():
      yield
    self._thread.join()
    return self.result()

  def join(self):
    """Wait for the underlying thread to complete."""
    return self._thread.join()

def _pytree_leaf_desc(leaf):
  if isinstance(leaf, (np.ndarray, jax.Array)):
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
  else:
    return leaf

def nonblocking_save(data: PyTreeT, directory: str | PathLike[str],
                     overwrite: bool = True, partial_write: bool = False,
                     tensorstore_specs: PyTreeT | None = None,
                     ) -> SerializationFuture:
  # start serialization immediately
  fut = SerializationFuture(save, data, directory, overwrite,
                            partial_write, tensorstore_specs)
  # construct a nice looking pytree representing the nodes being read
  fut.pytree = jax.tree.map(_pytree_leaf_desc, data)
  return fut


_is_desc_array = lambda x: (re.match(_ARRAY_TYPE_REGEX,
                                     x.split(_TYPE_ID_LEAF_DELIMITER, 1)[0])
                            is not None)
_none_is_leaf = lambda x: x is None

def nonblocking_load(directory: str | PathLike[str],
                     shardings: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
                     pytree: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
                     tensorstore_specs: PyTreeT | None = None,
                     best_effort: bool = False) -> SerializationFuture:
  if pytree is utils.MISSING:
    pytree = load_pytree(directory, best_effort=best_effort)

  # read in all the objects synchronously, but arrays asynchronously
  # use the existing partial-read functionality
  # the load here MUST BE CALLED BEFORE the async_load or it'll cause a deadlock
  # on multiprocess CPU machines (that need and XLA communication fallback)
  arr_pytree = jax.tree.map(lambda x: x if _is_desc_array(x) else None, pytree)
  obj_pytree = jax.tree.map(lambda x: None if _is_desc_array(x) else x, pytree)
  arr_shapes = jax.tree.map(_leaf_desc_to_leaf, arr_pytree)  # skip None-s here
  obj_data = load(directory, pytree=obj_pytree, best_effort=best_effort)
  pytree_stub = jax.tree.map(lambda x, y: x if x is not None else y,
                             arr_shapes, obj_data, is_leaf=_none_is_leaf)
  # TODO(rdyro): delete: pytree_stub = jax.tree.map(_leaf_desc_to_leaf, pytree)

  # TODO(rdyro): the output of this class is a workaround
  # it should return the fully populated pytree instead of just
  # jax.ShapeDtypeStruct for arrays by constructing them asynchronously
  fut = SerializationFuture(load, directory, shardings, pytree,
                            tensorstore_specs, best_effort)

  fut.pytree = pytree_stub
  return fut
