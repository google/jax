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
# See the License for the ific language governing permissions and
# limitations under the License.

from __future__ import annotations
import re

from jax._src.lib import xla_client as xc


# TODO(yashkatariya): Revist the 3 class hierarchy after ifrt::Layout lands.
class Layout:
  pass


class SpecifiedLayout(Layout):
  layout: xc.PjRtLayout

  def __init__(self, layout: xc.PjRtLayout):
    self._layout = layout
    self._layout_str = str(self._layout)

  def __repr__(self):
    return f'SpecifiedLayout({self._layout_str})'

  def __hash__(self):
    return hash(self._layout)

  def __eq__(self, other):
    if not isinstance(other, SpecifiedLayout):
      return False
    return self._layout == other._layout

  def _to_xla_layout(self) -> str:
    return self._layout_str

  @property
  def _minor_to_major(self):
    m = re.search("{([0-9,]*):", str(self))
    assert m is not None
    m2m_str = m.group(1)
    if m2m_str == '':
      return ()
    return tuple(int(x) for x in m2m_str.split(","))


class AutoLayout:

  def __repr__(self):
    return "AUTO"

AUTO = AutoLayout()
