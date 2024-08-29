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
"""Array serialization and deserialization."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Callable, Sequence
import functools
import itertools
import logging
import re
import threading
import time
from typing import Any

import jax
from jax._src import array
from jax._src import distributed
from jax._src import sharding
from jax._src import typing
from jax._src import util
from jax._src.layout import Layout
from jax._src.lib import xla_extension as xe
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
# pylint: disable=unused-import
# import tensorstore-backed methods for backward compatibility.
from jax.experimental.array_serialization.tensorstore_impl import (
    get_tensorstore_spec as _new_get_tensorstore_spec,
    run_deserialization, run_serialization,
    async_serialize, async_deserialize, TS_CONTEXT)
from jax.experimental.array_serialization.asyncio_utils import (
    _LimitInFlightBytes)
# pylint: enable=unused-import
import tensorstore as ts


_REMOVED_VALUE = 'Value removed'
_CHECKPOINT_SUCCESS = 'checkpoint_write_success'
_module_unique_count = itertools.count()
_DISTRIBUTED_SYSTEM_MSG = (
    'Please initialize the distributed system via '
    '`jax.distributed.initialize()` at the start of your program.')
_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_REMOTE_DRIVER_VALIDATIONS = [
    {'driver': 'gcs', 'path_regex': None},
    {'driver': 's3', 'path_regex': None},
]
_DEFAULT_DRIVER = 'file'

class BarrierTimeoutError(Exception):
  pass

_BARRIER_TIMED_OUT_MSG = (
    "Suggestions for possible fixes:\n"
    "* Check the logs to see if one or more processes failed.\n"
    "* Make sure the training and checkpointing endpoints are close geographically.\n"
    "* Try increasing the timeout you pass to GlobalAsyncCheckpointManager.")

logger = logging.getLogger(__name__)


def is_remote_storage(tspec: dict[str, Any] | str) -> bool:
  """Detect if user is using cloud storages.

  This can detect common defines and unable to detect some corner cases such as
  using gcsfuse.
  """
  if isinstance(tspec, str):
    # KvStoreUrl
    if re.match(rf'^({"|".join(_REMOTE_URL_PREFIXES)})', tspec):
      return True
    else:
      return False

  for key in ('base', 'kvstore'):
    if key in tspec:
      return is_remote_storage(tspec[key])

  if 'driver' in tspec:
    for rule in _REMOTE_DRIVER_VALIDATIONS:
      if tspec['driver'] == rule['driver']:
        if rule['path_regex'] is None:
          return True

        # check if path matches the regex.
        if re.match(rule['path_regex'], tspec['path']):
          return True

  return False

# for compatibility with older zarr format
_get_metadata = functools.partial(ts_impl. get_tensorstore_metadata,
                                  driver='zarr')
get_tensorstore_spec = functools.partial(ts_impl.get_tensorstore_spec,
                                         driver='zarr')

def _get_key(key: int):
  return f'tensorstore_checkpoint_{key}'


class GlobalAsyncCheckpointManagerBase(util.StrictABC):
  """Interface for checkpointing GDAs asynchronously.

  This class manages the state of an ongoing asynchronous checkpoint.

  For example, say a checkpoint happens on every step. If you checkpoint on
  step 1 and after some computation the model is on checkpoint 2. But step 1's
  checkpoint hasn't finished committing to the storage layer yet. So until that
  is finished, checkpoint for step 2 will need to be blocked. Maintaining a
  class allows to maintain that state.

  Examples:

  Below is a simplified training loop:

  ```
  # Call this at the start of your program.
  jax.distributed.initialize()

  manager = GlobalAsyncCheckpointManager()

  # Restore checkpoint if available or initialize the train_state from
  # init_fn().
  train_state = manager.deserialize(...)

  while ...:
    if step % num_steps_between_checkpoints == 0:
      manager.serialize(train_state, temp_checkpoint_dir=...,
                        final_checkpoint_dir=...)
      train_state = train_step(train_state, input)
      # This is a non-blocking call.
      manager.check_for_errors()

  manager.serialize(train_state, temp_checkpoint_dir=...,
                    final_checkpoint_dir=...)
  # Wait before the end of the program for the checkpoint to finish. This is a
  # blocking call.
  manager.wait_until_finished()
  ```
  """

  @abc.abstractmethod
  def check_for_errors(self):
    """Checks if any errors have been raised in the child thread.

    This is a non-blocking call that can be called in the main thread.
    """

  @abc.abstractmethod
  def wait_until_finished(self):
    """Blocks until serialization has finished."""

  @abc.abstractmethod
  def serialize(self, arrays, tensorstore_specs, *,
                on_commit_callback: Callable[[], None]):
    """Serializes GDAs to TensorStore."""

  @abc.abstractmethod
  def deserialize(self, shardings: Sequence[sharding.Sharding],
                  tensorstore_specs: Sequence[dict[str, Any]],
                  global_shapes: Sequence[array.Shape] | None = None,
                  dtypes: Sequence[typing.DTypeLike] | None = None):
    """Deserializes GDAs from TensorStore."""


class AsyncManager:

  def __init__(self, timeout_secs=300):
    self._timeout_secs = timeout_secs
    self._timeout_in_ms = self._timeout_secs * 1000

    self._commit_futures = None
    self._thread = None
    self._exception = None

    if jax.process_count() > 1 and distributed.global_state.client is None:
      raise ValueError(_DISTRIBUTED_SYSTEM_MSG)
    if jax.process_count() > 1:
      self._client = distributed.global_state.client
    self._count = None

  def __del__(self):
    if self._thread is not None and self._thread.is_alive():
      logger.warning('Please add `.wait_until_finished()` in the main thread '
                      'before your program finishes because there is a '
                      'possibility of losing errors raised if the '
                      'this class is deleted before writing is completed.')

  def _thread_func(self):
    try:
      current_process = jax.process_index()
      process_count = jax.process_count()
      logger.info('Starting commit to storage layer by process: %s',
                   current_process)
      thread_start_time = time.time()
      for future in self._commit_futures:
        future.result()
      logger.info('Finished committing to storage layer by process: %s',
                   current_process)

      key_for_barrier = None
      if process_count > 1:
        # All processes will wait at the barrier. When all processes are at the
        # barrier, the barrier will be satisfied. If not, then it will timeout.
        key_for_barrier = _get_key(self._count)
        logger.info('Key used for barrier is %s for process %s',
                    key_for_barrier, current_process)
        # pytype: disable=attribute-error
        self._client.wait_at_barrier(key_for_barrier, self._timeout_in_ms,
                                     process_ids=None)
        # pytype: enable=attribute-error
        logger.info('Finished waiting at barrier for process %s',
                    current_process)

      if current_process == 0:
        if self._on_commit_callback is not None:
          self._on_commit_callback()
          logger.info('on_commit_callback successfully ran!')
        if process_count > 1:
          # pytype: disable=attribute-error
          self._client.key_value_set(key_for_barrier, _CHECKPOINT_SUCCESS)
          # pytype: enable=attribute-error
          logger.info('Process 0 successfully set key %s in the kv store',
                      key_for_barrier)

      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/thread_duration_sec',
          time.time() - thread_start_time)

    except Exception as e:  # pylint: disable=broad-except
      self._exception = e

  def _start_async_commit(self, on_commit_callback):
    self._count = next(_module_unique_count)

    self._on_commit_callback = on_commit_callback
    self._thread = threading.Thread(target=self._thread_func)
    self._thread.start()

  def check_for_errors(self):
    if self._exception is not None:
      # Clears self._exception so it is only raised once.
      exception = self._exception
      self._exception = None
      if (isinstance(exception, xe.XlaRuntimeError) and
          'DEADLINE_EXCEEDED: Barrier timed out' in str(exception)):
        raise BarrierTimeoutError(
            '\n'.join([str(exception), _BARRIER_TIMED_OUT_MSG]))
      raise exception  # pylint: disable=raising-bad-type

  def wait_until_finished(self):
    if self._thread is not None:
      self._thread.join()
      self._thread = None
      logger.info('Thread joined successfully')

    self.check_for_errors()
    logger.info('Error check finished successfully')

    if jax.process_count() > 1 and self._count is not None:
      # Block until process 0 writes success value to the key value store.
      # If it fails to write it, then `blocking_key_value_get` will time out.
      get_key = _get_key(self._count)
      # pytype: disable=attribute-error
      self._client.blocking_key_value_get(get_key, self._timeout_in_ms)
      # pytype: enable=attribute-error
      logger.info('blocking_key_value_get on key %s was successfully '
                  'completed.', get_key)

  def _add_futures(self, futures: Sequence[asyncio.Future]):
    self._commit_futures = futures


class GlobalAsyncCheckpointManager(AsyncManager, GlobalAsyncCheckpointManagerBase):
  """Responsible for serializing GDAs via TensorStore."""

  def serialize(
      self,
      arrays,
      tensorstore_specs,
      *,
      on_commit_callback: Callable[[], None],
      transaction: ts.Transaction | None = None,
  ):
    """Serializes Arrays or Arrays via TensorStore asynchronously.

    TensorStore writes to a storage layer in 2 steps:
    *  Reading/copying from the source after which the source can be modified.
         * Returns a copy future.
    *  Writing/committing to the storage layer.
         * Returns a commit future.

    In asynchronous mode, the serialization waits for the commit future to
    finish in a separate thread allowing other computation to proceed.

    Args:
      arrays: Arrays or Arrays that should be serialized.
      tensorstore_specs: TensorStore specs that are used to serialize GDAs or
        Arrays.
      on_commit_callback: This callback will be executed after all processes
        have finished writing their checkpoints to disk. Filesystems where
        atomic rename operations are supported, you can rename from the
        temporary directory to the final directory. On GCS, you write to the
        final directory directly and in `on_commit_callback` you write a success
        file indicating that the serialization was successful because GCS does
        not support atomic rename operations.
      transaction: Optional TensorStore transaction to use.
    """
    logger.info('Waiting for previous serialization to finish.')
    self.wait_until_finished()

    commit_futures: list[ts.Future] = []

    async def _run_serializer():
      future_writer = jax.tree_util.tree_map(
          lambda arr_inp, tensorstore_spec: ts_impl.async_serialize(
              arr_inp,
              tensorstore_spec,
              commit_future=commit_futures,
              transaction=transaction,
          ),
          arrays,
          tensorstore_specs,
      )
      return await asyncio.gather(*future_writer)
    asyncio.run(_run_serializer())

    self._add_futures(commit_futures)

    # Used in wait_until_finished to check on process != 0, if the checkpoint
    # has finished writing.
    self._start_async_commit(on_commit_callback)

  def serialize_with_paths(
      self,
      arrays: Sequence[jax.Array],
      paths: Sequence[str],
      *,
      on_commit_callback: Callable[[], None] | None = None,
      transaction: ts.Transaction | None = None,
  ):
    tspecs = jax.tree.map(get_tensorstore_spec, paths)
    if on_commit_callback is None:
      on_commit_callback = lambda: None
    return self.serialize(
        arrays,
        tspecs,
        on_commit_callback=on_commit_callback,
        transaction=transaction,
    )

  def deserialize(self, shardings: Sequence[sharding.Sharding | Layout],
                  tensorstore_specs: Sequence[dict[str, Any]],
                  global_shapes: Sequence[array.Shape] | None = None,
                  dtypes: Sequence[typing.DTypeLike] | None = None,
                  concurrent_gb: int = 32):
    self.wait_until_finished()
    return ts_impl.run_deserialization(
        shardings, tensorstore_specs, global_shapes, dtypes, concurrent_gb)

  def deserialize_with_paths(
      self, shardings: Sequence[sharding.Sharding],
      paths: Sequence[str],
      global_shapes: Sequence[array.Shape] | None = None,
      dtypes: Sequence[typing.DTypeLike] | None = None,
      concurrent_gb: int = 32):
    tspecs = jax.tree.map(get_tensorstore_spec, paths)
    return self.deserialize(shardings, tspecs, global_shapes, dtypes,
                            concurrent_gb)
