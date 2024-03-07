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

import operator

from absl.testing import absltest

import jax
from jax._src import linear_util as lu
from jax._src import test_util as jtu
from jax._src import util

from jax._src.util import weakref_lru_cache
jax.config.parse_flags_with_absl()

try:
  from jax._src.lib import utils as jaxlib_utils
except:
  jaxlib_utils = None

class UtilTest(jtu.JaxTestCase):

  def test_wrapped_fun_transforms(self):
    """Test a combination of transforms."""

    def f(*args, **kwargs):
      """The function to be transformed.
      Scales the positional arguments by a factor.
      Takes only one keyword argument, the factor to scale by."""
      factor = kwargs.pop('factor', 2)  # For PY2
      assert not kwargs
      return tuple(a * factor for a in args)

    @lu.transformation_with_aux
    def kw_to_positional(factor, *args, **kwargs):
      """A transformation with auxiliary output.
      Turns all keyword parameters into positional ones.

      On entry, append the values of the keyword arguments to the positional
      arguments. On exit, take a list of results and recreate a dictionary
      from the tail of the results. The auxiliary output is the list of
      keyword keys.
      """
      kwargs_keys = kwargs.keys()
      new_args = tuple(kwargs[k] for k in kwargs_keys)
      new_kwargs = dict(factor=factor)
      results = yield args + new_args, new_kwargs  # Yield transformed (args, kwargs)
      # Assume results correspond 1:1 to the args + new_args
      assert len(results) == len(args) + len(new_args)
      aux_output = len(new_args)
      yield (results[0:len(args)],
             dict(zip(kwargs_keys, results[len(args):]))), aux_output

    wf = lu.wrap_init(f)  # Wraps `f` as a `WrappedFun`.
    wf, out_thunk = kw_to_positional(wf, 2)
    # Call the transformed function.
    scaled_positional, scaled_kwargs = wf.call_wrapped(1, 2, three=3, four=4)
    self.assertEqual((2, 4), scaled_positional)
    self.assertEqual(dict(three=6, four=8), scaled_kwargs)
    self.assertEqual(2, out_thunk())

  def test_weakref_lru_cache(self):
    @weakref_lru_cache
    def example_cached_fn(key):
      return object()

    class Key:
      def __init__(self):
        # Make a GC loop.
        self.ref_loop = [self]

    stable_keys = [Key() for _ in range(2049)]
    for i in range(10000):
      example_cached_fn(stable_keys[i % len(stable_keys)])
      example_cached_fn(Key())

  def test_weakref_lru_cache_asan_problem(self):

    @weakref_lru_cache
    def reference_loop_generator(x):
      return x

    for _ in range(4097):
      reference_loop_generator(lambda x: x)


class SafeMapTest(jtu.JaxTestCase):

  def test_safe_map(self):
    def unreachable(*args, **kwargs):
      raise RuntimeError("unreachable")

    self.assertEqual([], util.safe_map(unreachable, []))
    self.assertEqual([], util.safe_map(unreachable, (), []))
    self.assertEqual([], util.safe_map(unreachable, [], [], []))
    self.assertEqual([], util.safe_map(unreachable, [], iter([]), [], []))

    def double(x):
      return x * 2

    self.assertEqual([14], util.safe_map(double, (7,)))
    self.assertEqual([0, 2, 4, 6], util.safe_map(double, range(4)))

    def make_tuple(*args):
      return args

    self.assertEqual(
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        util.safe_map(make_tuple, range(4), range(4, 8)),
    )

  def test_safe_map_errors(self):
    with self.assertRaisesRegex(
        TypeError, "safe_map requires at least 2 arguments"
    ):
      util.safe_map()

    with self.assertRaisesRegex(
        TypeError, "safe_map requires at least 2 arguments"
    ):
      util.safe_map(lambda x: x)

    with self.assertRaisesRegex(TypeError, "'int' object is not callable"):
      util.safe_map(7, range(6))

    def error(*args, **kwargs):
      raise RuntimeError("hello")

    with self.assertRaisesRegex(RuntimeError, "hello"):
      util.safe_map(error, range(6))

    with self.assertRaisesRegex(
        ValueError, r"safe_map\(\) argument 2 is longer than argument 1"
    ):
      util.safe_map(operator.add, range(3), range(4))

    with self.assertRaisesRegex(
        ValueError, r"safe_map\(\) argument 2 is shorter than argument 1"
    ):
      util.safe_map(operator.add, range(7), range(2))

    with self.assertRaisesRegex(
        ValueError, r"safe_map\(\) argument 2 is longer than argument 1"
    ):
      util.safe_map(operator.add, (), range(3))


class SafeZipTest(jtu.JaxTestCase):

  def test_safe_zip(self):
    self.assertEqual([], util.safe_zip([]))
    self.assertEqual([], util.safe_zip((), []))
    self.assertEqual([], util.safe_zip([], [], []))
    self.assertEqual([], util.safe_zip([], iter([]), [], []))
    self.assertEqual([(7,)], util.safe_zip((7,)))
    self.assertEqual([(0,), (1,), (2,), (3,)], util.safe_zip(range(4)))
    self.assertEqual(
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        util.safe_zip(range(4), range(4, 8)),
    )

  def test_safe_zip_errors(self):
    with self.assertRaisesRegex(
        TypeError, "safe_zip requires at least 1 argument"
    ):
      util.safe_zip()

    with self.assertRaisesRegex(
        TypeError, "'function' object is not iterable"
    ):
      util.safe_zip(lambda x: x)

    with self.assertRaisesRegex(
        ValueError, r"safe_zip\(\) argument 2 is longer than argument 1"
    ):
      util.safe_zip(range(3), range(4))

    with self.assertRaisesRegex(
        ValueError, r"safe_zip\(\) argument 2 is shorter than argument 1"
    ):
      util.safe_zip(range(7), range(2))

    with self.assertRaisesRegex(
        ValueError, r"safe_zip\(\) argument 2 is longer than argument 1"
    ):
      util.safe_zip((), range(3))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
