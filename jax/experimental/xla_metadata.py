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

from __future__ import annotations

from typing import Any
import threading
from contextlib import contextmanager

from jax._src import core
from jax._src import config


class _XlaMetadata(threading.local):
  val: dict[Any, Any]

  def __init__(self):
    self.val = {}

thread_local_metadata = _XlaMetadata()

def current_xla_metadata():
  return thread_local_metadata.val

@contextmanager
def set_xla_metadata(**kwargs):
  new_metadata = thread_local_metadata.val.copy()
  new_metadata.update(**kwargs)
  prev_metadata, thread_local_metadata.val = (
      thread_local_metadata.val,
      new_metadata,
  )
  config.update_thread_local_jit_state(
      xla_metadata_ctx_manager=tuple(
          (v, k) for k, v in sorted(new_metadata.items())))
  try:
    yield
  finally:
    thread_local_metadata.val = prev_metadata
    config.update_thread_local_jit_state(
        xla_metadata_ctx_manager=tuple(
            (v, k) for k, v in prev_metadata.items()
        )
    )

core.register_eqn_context(
    "xla_metadata", core.EqnContextVal( current_xla_metadata, set_xla_metadata))
