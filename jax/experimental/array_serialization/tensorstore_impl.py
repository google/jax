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

import asyncio
from functools import partial
import functools
import os
from os import PathLike
import re
from typing import Any, Awaitable, Callable, Sequence
import math
import operator

import jax
from jax import numpy as jnp
from jax._src import array
from jax._src.layout import Layout
from jax._src import typing
from jax.experimental.array_serialization import asyncio_utils
import numpy as np
import tensorstore as ts

TS_ARRAY_DRIVER = "zarr3"

TS_CONTEXT = ts.Context({
  'file_io_concurrency': {'limit': 128},
  'cache_pool': {'total_bytes_limit': 10_000_000_000},  # 10 GB RAM limit
  'cache_pool#remote': {'total_bytes_limit': 10_000_000_000},
  'data_copy_concurrency': {'limit': 128}
})
# TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})
TS_CHUNK_LAYOUT = ts.ChunkLayout({
  "chunk": {"elements": 100_000_000},  # 100M (800MB for float64) file size
})

_DEFAULT_BASE_DRIVER = 'file'
_PROCESS_DIR_FORMAT = "process_{}"
_FILE_SIZE_TARGET = 2 * 1024 ** 3  # 2 GB

def _prod(x: Sequence[int]) -> int:
  return functools.reduce(operator.mul, x, 1)

def _maximum(x: Sequence[int], default: int = 1) -> list[int]:
  return [max(z, default) for z in x]

def is_tensorstore_spec_leaf(leaf: Any):
  # TODO(rdyro): think of a better way to detect which leaf is a ts config
  return isinstance(leaf, dict) and "driver" in leaf or "kvstore" in leaf

def _prime_factors(x: int) -> list[int]:
  factors = []
  while x % 2 == 0:
    factors.append(2)
    x //= 2
  for i in range(3, int(math.sqrt(x)) + 1, 2):
    while x % i == 0:
      factors.append(i)
      x //= i
  if x > 1:
    factors.append(x)
  return sorted(factors)

@functools.lru_cache(maxsize=1000)
def _compute_chunk_shape(
    local_shape: Sequence[int], dtype: str | jnp.dtype,
    file_size_target: int = _FILE_SIZE_TARGET) -> list[int]:
  """Compute a chunk such that divides local shape and is less than target file size."""
  # local_shape = _maximum(list(arr.addressable_data(0).shape), default=1)
  local_shape = _maximum(local_shape, default=1)
  total_size = (_prod(_maximum(local_shape, default=1))
                * jnp.dtype(dtype).itemsize)
  if len(local_shape) == 0:
    return local_shape
  axis_prime_factors = [_prime_factors(z) if z > 1 else [] for z in local_shape]
  chunk_size = total_size
  while chunk_size > 1.1 * file_size_target:  # 10% buffer
    chosen_axis_idx, chosen_divisor = None, 1
    for axis_idx in range(len(local_shape)):
      if len(axis_prime_factors[axis_idx]) == 1:
        continue
      if (chosen_axis_idx is None
          or chosen_divisor > axis_prime_factors[axis_idx][0]):
        chosen_axis_idx = axis_idx
        chosen_divisor = axis_prime_factors[axis_idx][0]
    if chosen_axis_idx is None:
      break
    if len(axis_prime_factors[chosen_axis_idx]) == 0:
      return local_shape
    prime_factors = axis_prime_factors[chosen_axis_idx]
    prime_factors: list[int]
    prime_factors.pop(0)
    local_shape[chosen_axis_idx] //= chosen_divisor
    chunk_size //= chosen_divisor
  return local_shape

def get_tensorstore_metadata(arr, is_remote: bool = False,
                             file_size_target: int = _FILE_SIZE_TARGET,
                             driver: str = TS_ARRAY_DRIVER,
                             ) -> dict[str, Any]:
  global_shape, dtype = arr.shape, arr.dtype
  if hasattr(arr, 'addressable_data'):
    local_shape = arr.addressable_data(0).shape
  else:
    local_shape = global_shape
  return _get_tensorstore_metadata(global_shape, dtype, local_shape, is_remote,
                                   file_size_target, driver)

@functools.lru_cache(maxsize=1000)
def _get_tensorstore_metadata(
  global_shape: Sequence[int], dtype: jnp.dtype, local_shape: Sequence[int],
  is_remote: bool = False, file_size_target: int = _FILE_SIZE_TARGET,
  driver: str = TS_ARRAY_DRIVER) -> dict[str, Any]:
  if driver == TS_ARRAY_DRIVER:
    codecs = ([{"name": "zstd"}] if is_remote else [])
    return {
        'codecs': codecs,
        'shape': global_shape,
        'data_type': jnp.dtype(dtype).name,
        'chunk_grid': {
          'name': 'regular',
          'configuration': {'chunk_shape': _compute_chunk_shape(
              local_shape, dtype, file_size_target=file_size_target)}
        }
    }
  elif driver == "zarr":  # in zarr dtype goes in the base spec
    return {'compressor': {'id': 'zstd'}, 'shape': global_shape,
            'chunks': np.array(np.maximum(1, local_shape))}
  else:
    raise ValueError(f"Unsupported driver: {driver}")

_divides = lambda x, y: np.all((np.array(x) % np.array(y)) == 0)

def merge_nested_specs(dict1: dict[Any, Any], dict2: dict[Any, Any]):
  """Merge two specs as nested dictionaries, dict2 takes precedence."""
  if dict2 is None:
    return dict1
  exclusive_dict1_keys = set(dict1.keys()) - set(dict2.keys())
  exclusive_dict2_keys = set(dict2.keys()) - set(dict1.keys())
  shared_keys = set(dict1.keys()) & set(dict2.keys())
  out_dict = {k: dict1[k] for k in exclusive_dict1_keys}
  out_dict.update({k: dict2[k] for k in exclusive_dict2_keys})
  for k in shared_keys:
    v1, v2 = dict1[k], dict2[k]
    if isinstance(v1, dict):
      out_dict[k] = merge_nested_specs(v1, v2)
    else:
      out_dict[k] = v2
  return out_dict

def verify_tensorstore_spec(spec: dict[str, Any], arr: jax.Array | None,
                            path: str | os.PathLike[str],
                            check_metadata: bool = True) -> None:
  """Verify the minimum requirements for a tensorstore spec."""
  if check_metadata:
    assert arr is not None, "Array is required for metadata verification."
    metadata = spec['metadata']
    msg = f"Provided dtype {metadata['data_type']} != array dtype: {arr.dtype}"
    assert metadata['data_type'] == jnp.dtype(arr.dtype).name, msg
    msg = f"Provided shape {metadata['shape']} != array shape: {arr.shape}"
    assert metadata['shape'] == arr.shape, msg
    local_shape = arr.addressable_data(0).shape
    chunk_shape = metadata['chunk_grid']['configuration']['chunk_shape']
    msg = (f"Provided chunk shape {chunk_shape} does not divide the local shape"
           f" of the array {local_shape}")
    assert _divides(local_shape, chunk_shape), msg
  # we don't support altering the path of the tensorstore
  msg = (f"Provided { path = } does not match the path in the spec:"
         f" {spec['kvstore']}")
  assert spec["kvstore"]['base']['path'] == str(path), msg

def _deprecated_get_metadata(arr):
  local_shape = arr.addressable_data(0).shape
  return {
      'compressor': {'id': 'zstd'},
      'shape': arr.shape,
      'chunks': np.array(np.maximum(1, local_shape)),
  }

def _spec_has_metadata(tree):
  if not isinstance(tree, dict):
    return False
  return 'metadata' in tree or any(
      _spec_has_metadata(subtree) for _, subtree in tree.items())

def _get_kvstore_for_gcs(ckpt_path: str):
  m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
  if m is None:
    raise ValueError('The ckpt_path should contain the bucket name and the '
                      f'file path inside the bucket. Got: {ckpt_path}')
  bucket = m.group(1)
  path_without_bucket = m.group(2)
  return {'driver': 'gcs', 'bucket': bucket, 'path': path_without_bucket}

def _get_kvstore_for_s3(ckpt_path: str):
  m = re.fullmatch('^s3://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
  if m is None:
    raise ValueError('The ckpt_path should contain the bucket name and the '
                      f'file path inside the bucket. Got: {ckpt_path}')
  bucket = m.group(1)
  path_without_bucket = m.group(2)
  return {'driver': 's3', 'bucket': bucket, 'path': path_without_bucket}

def get_tensorstore_spec(
    ckpt_path: str | PathLike[str], ocdbt: bool = False,
    process_num: int | None = None, arr: jax.Array | None = None,
    driver: str = TS_ARRAY_DRIVER) -> dict[str, Any]:

  # Normalize path to exclude trailing '/'. In GCS path case, we will need to
  # fix the path prefix to add back the stripped '/'.
  ckpt_path = str(ckpt_path)
  ckpt_path = re.sub(r"^gs:/", r"gs://", os.path.normpath(ckpt_path))
  ckpt_path = re.sub(r"^s3:/", r"s3://", ckpt_path)

  # in cases of multi-process writes, we need to write to a different location
  # for each process and finally created a combined symlink to the final
  # location, tensorstore can do this via ts.KvStore.experimental_copy_range_to
  if process_num is not None:
    _parent, _name = os.path.split(ckpt_path)
    ckpt_path = os.path.join(_parent, _PROCESS_DIR_FORMAT.format(process_num),
                             _name)

  is_gcs_path = ckpt_path.startswith('gs://')
  is_s3_path = ckpt_path.startswith('s3://')
  spec = {'driver': driver, 'kvstore': {}}

  # use a combined OCDBT store, the actual path is the parent path
  # the name (filename/last part of the path) is the key in the ocdbt kvstore
  entry_key = None
  if ocdbt:
    (ckpt_path, entry_key), org_ckpt_path = os.path.split(ckpt_path), ckpt_path
    if is_gcs_path:
      m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
    elif is_s3_path:
      m = re.fullmatch('^s4://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
    else:
      m = re.match("a", "a")  # make it True
    if m is None:
      raise ValueError('Using OCDBT requires the bucket name, the directory'
                       ' name and the array name, your path is: '
                       f'{org_ckpt_path}')

  if is_gcs_path:
    base_kvstore = _get_kvstore_for_gcs(ckpt_path)
  elif is_s3_path:
    base_kvstore = _get_kvstore_for_s3(ckpt_path)
  else:
    base_kvstore = {'driver': _DEFAULT_BASE_DRIVER, 'path': ckpt_path}

  if ocdbt:
    if not is_gcs_path and not os.path.isabs(ckpt_path):
      raise ValueError(f'Checkpoint path should be absolute. Got {ckpt_path}')
    spec['kvstore'] = {'driver': 'ocdbt', 'base': base_kvstore,
                       'path': entry_key}
  else:
    spec['kvstore'] = base_kvstore
  if arr is not None:
    spec["metadata"] = get_tensorstore_metadata(arr, driver=str(spec["driver"]))
  return spec

async def _create_async_array_from_callback(
    global_shape: array.Shape,
    inp_sharding: jax.sharding.Sharding,
    data_callback: Callable[[array.Index, jax.Device], Awaitable[jax.Array]],
):
  device_to_index_map = inp_sharding.devices_indices_map(global_shape)
  addressable_da = inp_sharding._addressable_device_assignment
  future_arrays = [data_callback(device_to_index_map[d], d)
                   for d in addressable_da]
  dbs = await asyncio.gather(*future_arrays)
  return array.make_array_from_single_device_arrays(
      global_shape, inp_sharding, dbs)

async def _transfer_shard_to_host(shard: array.Shard) -> np.ndarray:
  data = shard.data
  has_pinned_host = any(
      m.kind == "pinned_host" for m in shard.device.addressable_memories())
  if has_pinned_host:
    # If available, transfer to pinned host memory
    sharding = jax.sharding.SingleDeviceSharding(shard.device,
        memory_kind="pinned_host")
    data = jax.device_put(data, sharding)
  else:
    data.copy_to_host_async()
  # Allow other transfers to be scheduled simultaneously
  await asyncio.sleep(0)
  # Ensure that jax.Array's internal numpy array can be zero-copied. Tensorstore
  # implicitly converts the written data to a numpy array, and would otherwise
  # silently copy host-to-host.
  return np.array(data, copy=False)

async def combine_kvstores(combined_kvstore: dict[str, Any],
                           kvstores: list[dict[str, Any]],
                           context: ts.Context | dict[str, Any] = TS_CONTEXT
                           ) -> None:
  """Merge a list of kvstores into a single kvstore. NOT multi-process safe."""
  combined_fut = ts.KvStore.open(combined_kvstore, context=context)
  kvstores_futs = [ts.KvStore.open(kvstore, context=context)
                   for kvstore in kvstores]
  combined, kvstores = await asyncio.gather(combined_fut,
                                            asyncio.gather(*kvstores_futs))
  tx = ts.Transaction()
  await asyncio.gather(*[kvstore.experimental_copy_range_to(
      combined.with_transaction(tx)) for kvstore in kvstores])
  await tx.commit_async()

async def async_serialize(
    arr_inp,
    tensorstore_spec,
    commit_future=None,
    context=TS_CONTEXT,
    chunk_layout=TS_CHUNK_LAYOUT,
    primary_host: int | None = None,
    replica_id: int = 0,
    transaction: ts.Transaction | None = None,
):
  """Serialize an array using TensorStore.

  Args:
    arr_inp: The array to serialize.
    tensorstore_spec: The tensorstore spec to use.
    commit_future: A list of futures that will be appended to. The futures can
      be awaited asynchronously. If None, the futures will be awaited
      synchronously by this method.
    context: ts.Context instance.
    primary_host: Primary host, which indicates the host that will be treated as
      the "leader". If None, all hosts are treated as the primary. DO NOT USE
      unless you are sure you know what you are doing.
    replica_id: Allows overriding the shard replica id that will be saved. DO
      NOT USE unless you are sure you know what you are doing.
    transaction: TensorStore transaction to use for opening and writing the
      array.  If not specified, a non-transactional write will be used.
  """
  if (isinstance(arr_inp, array.ArrayImpl) and jax.process_count() > 1 and
      arr_inp.is_fully_addressable):
    raise ValueError(
        f'Passing fully addressable arrays to a multiprocess '
        f'serialization is not allowed, as this may lead to a race condition '
        f'between processes. Serialization have failed for the array with '
        f'the path from kvstore: "{tensorstore_spec["kvstore"]}".')

  # 'metadata' may not be present at the top level (for example, if we are using
  # a 'cast' driver).
  if not _spec_has_metadata(tensorstore_spec):
    tensorstore_spec['metadata'] = get_tensorstore_metadata(
        arr_inp, driver=tensorstore_spec['driver'])
  ## zarr driver requires specifying the dtype in the spec base
  if tensorstore_spec['driver'] == 'zarr' and 'dtype' not in tensorstore_spec:
    tensorstore_spec['dtype'] = jnp.dtype(arr_inp.dtype).name

  # If primary_host is None, all hosts will checkpoint. This is used
  # for checkpointing to local filesystem.
  if primary_host is None or jax.process_index() == primary_host:
    open_future = ts.open(
        ts.Spec(tensorstore_spec),
        create=True,
        open=True,
        context=context,
        chunk_layout=chunk_layout,
        transaction=transaction,
    )
    # Asynchronous case.
    if commit_future is not None:
      assert isinstance(commit_future, list)
      commit_future.append(open_future)
    else:
      await open_future

  # `ts.open` runs twice for process `primary_host` because for the first time,
  # we just get the future to be awaited upon in the background thread. The
  # second one runs with `assume_metadata=True` which does no I/O operation and
  # returns the tensorstore object.
  # For every process other than `primary_host`, we open with
  # `assume_metadata=True`.
  t = await ts.open(
      ts.Spec(tensorstore_spec),
      open=True,
      assume_metadata=True,
      context=context,
      chunk_layout=chunk_layout,
      transaction=transaction,
  )

  async def _write_array(shard):
    if shard.replica_id == replica_id:
      data = await _transfer_shard_to_host(shard)
      write_future = t[shard.index].write(
          data,
          # Avoid additional copy of input array into the TensorStore chunk
          # cache.  If `arr_inp` is a jax.Array, the result of converting
          # it to a NumPy array, as is done internally by TensorStore, is
          # guaranteed to be immutable and therefore it is safe to retain a
          # reference indefinitely.
          can_reference_source_data_indefinitely=isinstance(
              arr_inp, array.ArrayImpl
          ),
      )
      if commit_future is not None:
        assert isinstance(commit_future, list)
        commit_future.append(write_future.commit)
        await write_future.copy
      else:
        await write_future.commit

  local_shards = arr_inp.addressable_shards
  future_write_state = jax.tree_util.tree_map(_write_array, local_shards)
  return await asyncio.gather(*future_write_state)


def run_serialization(arrays, tensorstore_specs):
  async def _run_serializer():
    future_writer = jax.tree_util.tree_map(async_serialize, arrays, tensorstore_specs)
    return await asyncio.gather(*future_writer)
  asyncio.run(_run_serializer())


def estimate_read_memory_footprint(t: ts.TensorStore,
                                   domain: ts.IndexDomain) -> int:
  rank = t.rank
  num_bytes = t.dtype.numpy_dtype.itemsize
  chunk_template = t.chunk_layout.read_chunk_template
  if domain is None:
    domain = t.domain
  origin = domain.origin
  shape = domain.shape
  chunk_origin = chunk_template.origin
  chunk_shape = chunk_template.shape

  # Some TensorStore drivers are not chunked, e.g. the inline 'array' driver.
  # For those, instead of returning a near-infinite memory footprint, estimate
  # the footprint as the entire shape.
  for i in range(rank):
    if not chunk_template[i].finite:
      return domain.size * num_bytes

  # Otherwise, if we have a chunked driver, estimate based on chunk size.
  for i in range(rank):
    origin_value = origin[i]
    chunk_origin_value = chunk_origin[i]
    chunk_size = chunk_shape[i]
    lower = origin_value - chunk_origin_value
    upper = origin_value + shape[i] - chunk_origin_value
    lower_aligned = lower // chunk_size * chunk_size
    upper_aligned = -(-upper // chunk_size) * chunk_size
    num_bytes *= (upper_aligned - lower_aligned)

  return num_bytes


async def async_deserialize(
    user_in_sharding: jax.sharding.Sharding | Layout,
    tensorstore_spec: ts.Spec | dict[str, Any],
    global_shape: Sequence[int] | None = None,
    dtype=None,
    byte_limiter: asyncio_utils._LimitInFlightBytes | None = None,
    context=TS_CONTEXT,
    chunk_layout=TS_CHUNK_LAYOUT,
    assume_metadata: bool = False,
):
  in_sharding = (user_in_sharding.sharding
                 if isinstance(user_in_sharding, Layout) else user_in_sharding)
  if not isinstance(in_sharding, jax.sharding.Sharding):
    raise ValueError(
        'sharding passed to deserialization should be specified, concrete and'
        f' an instance of `jax.sharding.Sharding`. Got {in_sharding}')
  dll = (user_in_sharding.device_local_layout
         if isinstance(user_in_sharding, Layout) else None)
  t = await ts.open(
      tensorstore_spec,
      open=True,
      assume_metadata=assume_metadata,
      context=context,
      chunk_layout=chunk_layout,
  )
  shape = t.shape if global_shape is None else global_shape
  new_shard_shape = in_sharding.shard_shape(tuple(shape))

  async def cb(index: array.Index, device: jax.Device):
    requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
    restricted_domain = t.domain.intersect(requested_domain)
    requested_bytes = estimate_read_memory_footprint(t, restricted_domain)
    # Limit the bytes read for every shard.
    if byte_limiter is not None:
      await byte_limiter.wait_for_bytes(requested_bytes)
    # This maybe needed because the shape the array was saved with is smaller
    # than the requested shape of the array in which it will be reloaded. So
    # the extra values will be filled with 0s.
    out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
    await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][
        restricted_domain].write(t[restricted_domain])
    if dtype is not None:
      # Cast while reloading on process to avoid 2 copies on device if the
      # casting is done on device.
      out = out.astype(dtype)
    # Convert to jnp array so that layouts are initialized properly for
    # sub-byte dtypes.
    # TODO(yashkatariya): This is a band-aid fix. Figure out a better way to
    # make this work.
    if out.dtype == jnp.int4:
      out = jnp.asarray(out)  # type: ignore
    result = jax.device_put(
        out, Layout(dll, jax.sharding.SingleDeviceSharding(device)))
    if byte_limiter is not None:
      # NB: `out` actually might not be ready for garbage collection by the
      # time we call release_bytes . Thus peak memory usage still might grow
      # beyond what byte_limiter limit suggests it should. The simplest option
      # would be to call  `result.block_until_ready()`` here. However it
      # also comes with ~15-20% perf penalty as we would be waiting for CPU->GPU
      # transfer instead of loading data. In the future, if memory pressure
      # becomes a problem, we can instead instrument  bytelimiter to
      # keep track of all in-flight tensors and only block_until_ready, if byte
      # limiter hits the limit to get reduced memory usage, without losing
      # performance in common use cases.
      await byte_limiter.release_bytes(requested_bytes)
    return result

  return await _create_async_array_from_callback(tuple(shape), in_sharding, cb)


def run_deserialization(shardings: Sequence[jax.sharding.Sharding | Layout],
                        tensorstore_specs: Sequence[dict[str, Any]],
                        global_shapes: Sequence[array.Shape] | None = None,
                        dtypes: Sequence[typing.DTypeLike] | None = None,
                        concurrent_gb: int = 32):
  concurrent_bytes = concurrent_gb * 10**9

  async def _run_deserializer():
    # Object should be created once per process.
    byte_limiter = asyncio_utils._LimitInFlightBytes(concurrent_bytes)

    future_arrays = jax.tree_util.tree_map(
        partial(async_deserialize, byte_limiter=byte_limiter),
        shardings, tensorstore_specs,
        [None] * len(tensorstore_specs) if global_shapes is None else global_shapes,
        [None] * len(tensorstore_specs) if dtypes is None else dtypes)
    return await asyncio.gather(*future_arrays)
  return asyncio.run(_run_deserializer())
