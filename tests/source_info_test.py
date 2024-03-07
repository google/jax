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
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import inspect

from absl.testing import absltest

import jax
from jax import lax
from jax._src import source_info_util
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class SourceInfoTest(jtu.JaxTestCase):

  def test_inline_jit_location_uses_callee_location(self):

    # 'f' should be inlined into both 'g' and 'h', using the source line
    # information of the call site. In particular, the source line information
    # of 'h' should not refer to the source information of 'g'.
    @partial(jax.jit, inline=True)
    def f(x): return lax.add(x, 3)

    def g(x): return lax.add(f(x), 4)

    def h(x): return lax.add(f(x), 5)

    for fn in (g, h):
      lines, fn_startline = inspect.getsourcelines(fn)
      fn_endline = fn_startline + len(lines)
      jaxpr = jax.make_jaxpr(fn)(2)
      for eqn in jaxpr.eqns:
        frame = source_info_util.user_frame(eqn.source_info)
        assert frame is not None, eqn
        self.assertLessEqual(fn_startline, frame.start_line)
        self.assertLessEqual(frame.end_line, fn_endline)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
