# Copyright 2018 The JAX Authors.
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
from functools import partial
import itertools
import typing
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import lax
from jax import numpy as jnp
from jax import ops

from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src import util
from jax._src.lax import lax as lax_internal
from jax._src.util import NumpyComplexWarning

config.parse_flags_with_absl()

# We disable the whitespace continuation check in this file because otherwise it
# makes the test name formatting unwieldy.
# pylint: disable=bad-continuation


ARRAY_MSG = r"Using a non-tuple sequence for multidimensional indexing is not allowed.*arr\[array\(seq\)\]"
TUPLE_MSG = r"Using a non-tuple sequence for multidimensional indexing is not allowed.*arr\[tuple\(seq\)\]"


float_dtypes = jtu.dtypes.floating
default_dtypes = float_dtypes + jtu.dtypes.integer
all_dtypes = default_dtypes + jtu.dtypes.boolean

class IndexSpec(typing.NamedTuple):
  shape: tuple[int, ...]
  indexer: Any
  out_shape: tuple[int, ...] | None = None


def check_grads(f, args, order, atol=None, rtol=None, eps=None):
  # TODO(mattjj,dougalm): add higher-order check
  default_tol = 1e-6 if config.enable_x64.value else 1e-2
  atol = atol or default_tol
  rtol = rtol or default_tol
  eps = eps or default_tol
  jtu.check_jvp(f, partial(jax.jvp, f), args, atol, rtol, eps)
  jtu.check_vjp(f, partial(jax.vjp, f), args, atol, rtol, eps)


STATIC_INDEXING_TESTS = [
  ("OneIntIndex", [
    IndexSpec(shape=(3,), indexer=1, out_shape=()),
    IndexSpec(shape=(3, 3), indexer=0, out_shape=(3,)),
    IndexSpec(shape=(3, 4, 5), indexer=2, out_shape=(4, 5)),
    IndexSpec(shape=(3,), indexer=-1, out_shape=()),
    IndexSpec(shape=(3,), indexer=-2, out_shape=()),
  ]),
  ("TwoIntIndices", [
    IndexSpec(shape=(3, 3), indexer=(2, 1), out_shape=()),
    IndexSpec(shape=(3, 4, 5), indexer=(1, 2), out_shape=(5,)),
    IndexSpec(shape=(3, 4, 5), indexer=(-1, 2), out_shape=(5,)),
  ]),
  ("ThreeIntIndices", [
    IndexSpec(shape=(3, 4, 5), indexer=(1, 2, 3), out_shape=()),
  ]),
  ("OneSliceIndex", [
    IndexSpec(shape=(10,), indexer=slice(1, 3), out_shape=(2,)),
    IndexSpec(shape=(10,), indexer=slice(1, -1), out_shape=(8,)),
    IndexSpec(shape=(10,), indexer=slice(None, -1), out_shape=(9,)),
    IndexSpec(shape=(10,), indexer=slice(None, None, None), out_shape=(10,)),
    IndexSpec(shape=(10, 8), indexer=slice(1, 3), out_shape=(2, 8)),
    IndexSpec(shape=(10, 8), indexer=slice(1, None), out_shape=(9, 8)),
    IndexSpec(shape=(10, 8), indexer=slice(None, 3), out_shape=(3, 8)),
    IndexSpec(shape=(10, 8), indexer=slice(-3, None), out_shape=(3, 8)),
  ]),
  ("OneSliceIndexNegativeStride", [
    IndexSpec(shape=(10,), indexer=slice(3, 1, -1), out_shape=(2,)),
    IndexSpec(shape=(10,), indexer=slice(1, 8, -1), out_shape=(0,)),
    IndexSpec(shape=(10,), indexer=slice(None, 1, -2), out_shape=(4,)),
    IndexSpec(shape=(10,), indexer=slice(None, None, -1), out_shape=(10,)),
    IndexSpec(shape=(10, 8), indexer=slice(3, 1, -1), out_shape=(2, 8)),
    IndexSpec(shape=(10, 8), indexer=slice(0, 8, -1), out_shape=(0, 8)),
    IndexSpec(shape=(10, 8), indexer=slice(None, None, -1), out_shape=(10, 8)),
  ]),
  ("SliceIndexClamping", [
    IndexSpec(shape=(10,), indexer=slice(2, 11, 1), out_shape=(8,)),
    IndexSpec(shape=(10,), indexer=slice(11, 12, 1), out_shape=(0,)),
    IndexSpec(shape=(10,), indexer=slice(-11, -2, 1), out_shape=(8,)),
    IndexSpec(shape=(10,), indexer=slice(-2, -12, -1), out_shape=(9,)),
    IndexSpec(shape=(10,), indexer=slice(12, -12, -1), out_shape=(10,)),
  ]),
  ("OneSliceIndexNonUnitStride", [
    IndexSpec(shape=(10,), indexer=slice(0, 8, 2), out_shape=(4,)),
    IndexSpec(shape=(10,), indexer=slice(0, 8, 3), out_shape=(3,)),
    IndexSpec(shape=(10,), indexer=slice(1, 3, 2), out_shape=(1,)),
    IndexSpec(shape=(10,), indexer=slice(1, None, 2), out_shape=(5,)),
    IndexSpec(shape=(10,), indexer=slice(None, 1, -2), out_shape=(4,)),
    IndexSpec(shape=(10, 8), indexer=slice(1, 8, 3), out_shape=(3, 8)),
    IndexSpec(shape=(10, 8), indexer=slice(None, None, 2), out_shape=(5, 8)),
    IndexSpec(shape=(10, 8), indexer=slice(None, 1, -2), out_shape=(4, 8)),
    IndexSpec(shape=(10, 8), indexer=slice(None, None, -2), out_shape=(5, 8)),
  ]),
  ("TwoSliceIndices", [
    IndexSpec(shape=(10, 8), indexer=(slice(1, 3), slice(0, 2)),
              out_shape=(2, 2)),
    IndexSpec(shape=(10, 8), indexer=(slice(1, None), slice(None, 2)),
              out_shape=(9, 2)),
    IndexSpec(shape=(10, 8), indexer=(slice(None, None, -1), slice(None, 2)),
              out_shape=(10, 2)),
    IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, 2)),
              out_shape=(2, 2, 3)),
    IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, None)),
              out_shape=(2, 8, 3)),
    IndexSpec(shape=(10, 8, 3), indexer=(slice(1, None), slice(0, 2)),
              out_shape=(9, 2, 3)),
  ]),
  ("OneColonIndex", [
    IndexSpec(shape=(3,), indexer=slice(None), out_shape=(3,)),
    IndexSpec(shape=(3, 4), indexer=slice(None), out_shape=(3, 4)),
  ]),
  ("MultipleColonIndices", [
    IndexSpec(shape=(3, 4), indexer=(slice(None), slice(None)),
              out_shape=(3, 4)),
    IndexSpec(shape=(3, 4, 5), indexer=(slice(None), slice(None)),
              out_shape=(3, 4, 5)),
  ]),
  ("MixedSliceIndices", [
    IndexSpec(shape=(10, 4), indexer=(slice(None), slice(0, 2)),
              out_shape=(10, 2)),
    IndexSpec(shape=(10, 4), indexer=(1, slice(None)),
              out_shape=(4,)),
  ]),
  ("EllipsisIndex", [
    IndexSpec(shape=(3,), indexer=Ellipsis, out_shape=(3,)),
    IndexSpec(shape=(3, 4), indexer=Ellipsis, out_shape=(3, 4)),
    IndexSpec(shape=(3, 4, 5), indexer=(0, Ellipsis), out_shape=(4, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis, 2, 3), out_shape=(3,)),
  ]),
  ("NoneIndex", [
    IndexSpec(shape=(), indexer=None, out_shape=(1,)),
    IndexSpec(shape=(), indexer=(None, None), out_shape=(1, 1)),
    IndexSpec(shape=(), indexer=(Ellipsis, None), out_shape=(1,)),
    IndexSpec(shape=(3,), indexer=None, out_shape=(1, 3)),
    IndexSpec(shape=(3, 4), indexer=None, out_shape=(1, 3, 4)),
    IndexSpec(shape=(3, 4), indexer=(Ellipsis, None), out_shape=(3, 4, 1)),
    IndexSpec(shape=(3, 4), indexer=(0, None, Ellipsis), out_shape=(1, 4)),
    IndexSpec(shape=(3, 4, 5), indexer=(1, None, Ellipsis), out_shape=(1, 4, 5)),
  ]),
  ("EmptyIndex", [
    IndexSpec(shape=(), indexer=(), out_shape=()),
    IndexSpec(shape=(3,), indexer=(), out_shape=(3,)),
    IndexSpec(shape=(3, 4), indexer=(), out_shape=(3, 4)),
  ]),
  ("TupleOfIntAndSliceAndIntArray", [
    IndexSpec(shape=(3, 2, 3), indexer=(0, slice(None), np.arange(3)),
              out_shape=(3, 2)),
    IndexSpec(shape=(3, 2, 3), indexer=(np.int32(1), slice(None), np.arange(3)),
              out_shape=(3, 2)),
    IndexSpec(shape=(3, 2, 3), indexer=(np.array(2), slice(None), np.arange(3)),
              out_shape=(3, 2)),
  ]),
]

STATIC_INDEXING_OUT_OF_BOUNDS_TESTS = [
  ("OneIntIndex", [
      IndexSpec(shape=(3,), indexer=-4, out_shape=()),
      IndexSpec(shape=(3, 3), indexer=3, out_shape=(3,)),
      IndexSpec(shape=(3, 4, 5), indexer=4, out_shape=(4, 5)),
  ]),
  ("TwoIntIndices", [
      IndexSpec(shape=(3, 3), indexer=(2, -4), out_shape=()),
      IndexSpec(shape=(3, 4, 5), indexer=(3, 2), out_shape=()),
      IndexSpec(shape=(3, 4, 5), indexer=(-4, 4), out_shape=(5,)),
  ]),
]


ADVANCED_INDEXING_TESTS = [
  ("One1DIntArrayIndex", [
    IndexSpec(shape=(3,), indexer=np.array([0, 1]), out_shape=(2,)),
    IndexSpec(shape=(3, 3), indexer=np.array([1, 2, 1]), out_shape=(3, 3)),
    IndexSpec(shape=(3, 4, 5), indexer=np.array([0, 2, 0, 1]),
              out_shape=(4, 4, 5)),
    IndexSpec(shape=(3,), indexer=np.array([-1,  1]), out_shape=(2,)),
    IndexSpec(shape=(3,), indexer=np.array([-2, -1]), out_shape=(2,)),
    IndexSpec(shape=(0,), indexer=np.array([], dtype=np.int32),
              out_shape=(0,)),
  ]),
  ("One2DIntArrayIndex", [
    IndexSpec(shape=(3,), indexer=np.array([[0, 0]]),out_shape=(1, 2)),
    IndexSpec(shape=(3, 3), indexer=np.array([[1, 2, 1], [0, 1, -1]]),
              out_shape=(2, 3, 3)),
    IndexSpec(shape=(3, 4, 5), indexer=np.array([[0, 2, 0, 1], [-1, -2, 1, 0]]),
              out_shape=(2, 4, 4, 5)),
  ]),
  ("Two1DIntArrayIndicesNoBroadcasting", [
    IndexSpec(shape=(3, 3), indexer=(np.array([0, 1]), np.array([1, 2])),
              out_shape=(2,)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([0, 2, 0, 1]), np.array([-1, 0, -1, 2])),
              out_shape=(4, 5)),
  ]),
  ("Two1DIntArrayIndicesWithBroadcasting", [
    IndexSpec(shape=(3, 3), indexer=(np.array([[0, 1]]), np.array([1, 2])),
              out_shape=(1, 2)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([[0, 2, 0, 1]]), np.array([-1, 0, -1, 2])),
              out_shape=(1, 4, 5)),
  ]),
  ("ArrayOfInts", [
    IndexSpec(shape=(3,), indexer=np.array([0, 1, 0]), out_shape=(3,)),
    IndexSpec(shape=(3, 4, 5), indexer=np.array([ 0, -1]), out_shape=(2, 4, 5)),
  ]),
  ("TupleOfListsOfPythonInts", [
    IndexSpec(shape=(3, 4, 5), indexer=([0, 1],), out_shape=(2, 4, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0, 3]]),
              out_shape=(2, 4, 5)),
  ]),
  ("TupleOfPythonIntsAndIntArrays", [
    IndexSpec(shape=(3, 4, 5), indexer=(0, np.array([0, 1])), out_shape=(2, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=(0, 1, np.array([[2, 3, 0, 3]])),
              out_shape=(1, 4)),
  ]),
  ("TupleOfListsOfPythonIntsAndIntArrays", [
    IndexSpec(shape=(3, 4, 5), indexer=([0, 1], np.array([0])),
              out_shape=(2, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], np.array([[2, 3, 0, 3]])),
              out_shape=(2, 4, 5)),
  ]),
]

ADVANCED_INDEXING_TESTS_NO_REPEATS = [
  ("One1DIntArrayIndex", [
    IndexSpec(shape=(3,), indexer=np.array([0, 1]), out_shape=(2,)),
    IndexSpec(shape=(3, 3), indexer=np.array([1, 2, 0]), out_shape=(3, 3)),
    IndexSpec(shape=(3, 4, 5), indexer=np.array([0, 2, 1]),
              out_shape=(3, 4, 5)),
    IndexSpec(shape=(3,), indexer=np.array([-1,  1]), out_shape=(2,)),
    IndexSpec(shape=(3,), indexer=np.array([-2, -1]), out_shape=(2,)),
    IndexSpec(shape=(0,), indexer=np.array([], dtype=np.int32), out_shape=(0,)),
  ]),
  ("One2DIntArrayIndex", [
    IndexSpec(shape=(3,), indexer=np.array([[0, 1]]), out_shape=(1, 2)),
    IndexSpec(shape=(6, 6), indexer=np.array([[1, 2, 0], [3, 4, -1]]),
              out_shape=(2, 3, 6)),
  ]),
  ("Two1DIntArrayIndicesNoBroadcasting", [
    IndexSpec(shape=(3, 3), indexer=(np.array([0, 1]), np.array([1, 2])),
              out_shape=(2,)),
    IndexSpec(shape=(4, 5, 6),
              indexer=(np.array([0, 2, 1, 3]), np.array([-1, 0, -2, 1])),
              out_shape=(4, 6)),
  ]),
  ("Two1DIntArrayIndicesWithBroadcasting", [
    IndexSpec(shape=(3, 3), indexer=(np.array([[0, 1]]), np.array([1, 2])),
              out_shape=(1, 2)),
    IndexSpec(shape=(4, 5, 6),
              indexer=(np.array([[0, 2, -1, 1]]), np.array([-1, 0, -2, 2])),
              out_shape=(1, 4, 6)),
  ]),
  ("ArrayOfInts", [
    IndexSpec(shape=(3,), indexer=np.array([0, 2, 1]), out_shape=(3,)),
    IndexSpec(shape=(3, 4, 5), indexer=np.array([ 0, -1]), out_shape=(2, 4, 5)),
  ]),
  ("TupleOfListsOfPythonInts", [
    IndexSpec(shape=(3, 4, 5), indexer=([0, 1],), out_shape=(2, 4, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0]]),
              out_shape=(2, 3, 5)),
  ]),
  ("TupleOfPythonIntsAndIntArrays", [
    IndexSpec(shape=(3, 4, 5), indexer=(0, np.array([0, 1])), out_shape=(2, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=(0, 1, np.array([[2, 3, 0]])),
              out_shape=(1, 3)),
  ]),
  ("TupleOfListsOfPythonIntsAndIntArrays", [
    IndexSpec(shape=(3, 4, 5), indexer=([0, 1], np.array([0])),
              out_shape=(2, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], np.array([[2, 3, 0]])),
              out_shape=(2, 3, 5)),
  ]),
]

ADVANCED_INDEXING_TESTS_NO_REPEATS_SORTED = [
  ("One1DIntArrayIndex", [
    IndexSpec(shape=(3,), indexer=np.array([0, 1]), out_shape=(2,)),
    IndexSpec(shape=(3, 3), indexer=np.array([0, 1, 2]), out_shape=(3, 3)),
    IndexSpec(shape=(3, 4, 5), indexer=np.array([0, 1, 2]),
              out_shape=(3, 4, 5)),
    IndexSpec(shape=(3,), indexer=np.array([-1,  1]), out_shape=(2,)),
    IndexSpec(shape=(3,), indexer=np.array([-2, -1]), out_shape=(2,)),
    IndexSpec(shape=(0,), indexer=np.array([], dtype=np.int32), out_shape=(0,)),
  ]),
  ("One2DIntArrayIndex", [
    IndexSpec(shape=(3,), indexer=np.array([[0, 1]]), out_shape=(1, 2)),
    IndexSpec(shape=(6, 6), indexer=np.array([[-1,  0,  1],
       [ 2,  3,  4]]), out_shape=(2, 3, 6)),
  ]),
  ("Two1DIntArrayIndicesNoBroadcasting", [
    IndexSpec(shape=(3, 3), indexer=(np.array([0, 1]), np.array([1, 2])),
              out_shape=(2,)),
    IndexSpec(shape=(4, 5, 6),
              indexer=(np.array([0, 1, 2, 3]), np.array([-2, -1,  0,  1])),
              out_shape=(4, 6)),
  ]),
  ("Two1DIntArrayIndicesWithBroadcasting", [
    IndexSpec(shape=(3, 3), indexer=(np.array([[0, 1]]), np.array([1, 2])),
              out_shape=(1, 2)),
    IndexSpec(shape=(4, 5, 6),
              indexer=(np.array([[-1, 0, 1, 2]]), np.array([-2, -1, 0, 2])),
              out_shape=(1, 4, 6)),
  ]),
  ("TupleOfListsOfPythonInts", [
    IndexSpec(shape=(3, 4, 5), indexer=([0, 1],), out_shape=(2, 4, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[0, 2, 3]]),
              out_shape=(2, 3, 5)),
  ]),
  ("TupleOfPythonIntsAndIntArrays", [
    IndexSpec(shape=(3, 4, 5), indexer=(0, np.array([0, 1])), out_shape=(2, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=(0, 1, np.array([[0, 2, 3]])),
              out_shape=(1, 3)),
  ]),
  ("TupleOfListsOfPythonIntsAndIntArrays", [
    IndexSpec(shape=(3, 4, 5), indexer=([0, 1], np.array([0])),
              out_shape=(2, 5)),
    IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], np.array([[0, 2, 3]])),
              out_shape=(2, 3, 5)),
  ]),
]


MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS = [
  ("SlicesAndOneIntArrayIndex", [
    IndexSpec(shape=(2, 3), indexer=(np.array([0, 1]), slice(1, 2)),
              out_shape=(2, 1)),
    IndexSpec(shape=(2, 3), indexer=(slice(0, 2), np.array([0, 2])),
              out_shape=(2, 2)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(Ellipsis, np.array([0, 2]), slice(None)),
              out_shape=(3, 2, 5)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(Ellipsis, np.array([[0, 2], [1, 3]]), slice(None)),
              out_shape=(3, 2, 2, 5)),
  ]),
  ("SlicesAndTwoIntArrayIndices", [
    IndexSpec(shape=(3, 4, 5),
              indexer=(Ellipsis, np.array([0, 2]), np.array([-1, 2])),
              out_shape=(3, 2)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([0, 2]), Ellipsis, np.array([-1, 2])),
              out_shape=(2, 4)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([0, 2]), np.array([-1,  2]), Ellipsis),
              out_shape=(2, 5)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([0, 2]), np.array([-1,  2]), slice(1, 3)),
              out_shape=(2, 2)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([0, 2]), slice(1, 3), np.array([-1,  2])),
              out_shape=(2, 2)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([ 0,  2, -2]), slice(None, None, 2),
                       np.array([-1,  2,  1])),
              out_shape=(3, 2)),
  ]),
  ("NonesAndIntArrayIndices", [
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([0, 2]), None, np.array([-1,  2])),
              out_shape=(2, 1, 5)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([0, 2]), None, None, np.array([-1,  2])),
              out_shape=(2, 1, 1, 5)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(Ellipsis, np.array([0, 2]), None, None,
                       np.array([-1,  2])),
              out_shape=(2, 3, 1, 1)),
  ]),
  ("IntArrayWithInt32Type", [
    IndexSpec(shape=(3, 4), indexer=(Ellipsis, np.array(1, dtype=np.int32)),
              out_shape=(3,)),
  ]),
]


MIXED_ADVANCED_INDEXING_TESTS = MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS + [
  ("SlicesAndOneIntArrayIndex", [
    IndexSpec(shape=(3, 4, 5),
              indexer=(Ellipsis, np.array([[0, 2], [1, 1]]), slice(None)),
              out_shape=(3, 2, 2, 5)),
  ]),
  ("SlicesAndTwoIntArrayIndices", [
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([ 0, 2, -2]), slice(None, None, 2),
                       np.array([-1, 2, -1])),
              out_shape=(3, 2)),
    IndexSpec(shape=(3, 4, 5),
              indexer=(np.array([[0, 2], [2, 0]]), Ellipsis,
                       np.array([[1, 0], [1, 0]])),
              out_shape=(2, 2, 4)),
  ]),
]

MODES = ["clip", "drop", "promise_in_bounds"]


class IndexingTest(jtu.JaxTestCase):
  """Tests for Numpy indexing translation rules."""

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer)
     for name, index_specs in STATIC_INDEXING_TESTS
     for shape, indexer, _ in index_specs],
    dtype=all_dtypes
  )
  def testStaticIndexing(self, name, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda x: np.asarray(x)[indexer]
    jnp_fun = lambda x: jnp.asarray(x)[indexer]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)
    # Tests x.at[...].get(...) as well.
    jnp_fun = lambda x: jnp.asarray(x).at[indexer].get()
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testStaticIndexingWithJaxArray(self):
    shape = (10,)
    indexer = slice(jnp.array(2, dtype=np.int32),
                    np.array(11, dtype=np.int32),
                    jnp.array(1, dtype=np.int32))
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, np.int32)]
    np_fun = lambda x: np.asarray(x)[indexer]
    jnp_fun = lambda x: jnp.asarray(x)[indexer]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)
    # Tests x.at[...].get(...) as well.
    jnp_fun = lambda x: jnp.asarray(x).at[indexer].get()
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    funcname=["negative", "sin", "cos", "square", "sqrt", "log", "exp"],
  )
  def testIndexApply(self, funcname, size=10, dtype='float32'):
    rng = jtu.rand_default(self.rng())
    idx_rng = jtu.rand_int(self.rng(), -size, size)
    np_func = getattr(np, funcname)
    jnp_func = getattr(jnp, funcname)
    @jtu.ignore_warning(category=RuntimeWarning)
    def np_op(x, idx):
      y = x.copy()
      np_func.at(y, idx)
      return y
    def jnp_op(x, idx):
      return jnp.asarray(x).at[idx].apply(jnp_func)

    # Test with traced integer index
    args_maker = lambda: [rng(size, dtype), idx_rng(size, int)]
    tol = (
        5e-5
        if jtu.test_device_matches(["tpu"]) and funcname in ("log", "exp")
        else None
    )
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker, atol=tol)
    self._CompileAndCheck(jnp_op, args_maker)

    # Test with slice index
    idx = slice(1, 5)
    np_op_idx = partial(np_op, idx=idx)
    jnp_op_idx = partial(jnp_op, idx=idx)
    args_maker = lambda: [rng(size, dtype)]
    self._CheckAgainstNumpy(np_op_idx, jnp_op_idx, args_maker, atol=tol,
                            rtol=tol)
    self._CompileAndCheck(jnp_op_idx, args_maker)

  def testIndexApplyBatchingBug(self):
    # https://github.com/google/jax/issues/16655
    arr = jnp.array([[1, 2, 3, 4, 5, 6]])
    ind = jnp.array([3])
    func = lambda a, i: a.at[i].apply(lambda x: x - 1)
    expected = jnp.array(list(map(func, arr, ind)))
    out = jax.vmap(func)(arr, ind)
    self.assertArraysEqual(out, expected)

  def testIndexUpdateScalarBug(self):
    # https://github.com/google/jax/issues/14923
    a = jnp.arange(10.)
    out = a.at[0].apply(jnp.cos)
    self.assertArraysEqual(out, a.at[0].set(1))

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer, mode=mode)
     for mode in MODES
     for name, index_specs in (
       STATIC_INDEXING_TESTS if mode == "promise_in_bounds" else
         STATIC_INDEXING_TESTS + STATIC_INDEXING_OUT_OF_BOUNDS_TESTS)
     for shape, indexer, _ in index_specs
    ],
    dtype=float_dtypes,
  )
  def testStaticIndexingGrads(self, name, shape, dtype, indexer, mode):
    rng = jtu.rand_default(self.rng())
    tol = 1e-2 if jnp.finfo(dtype).bits == 32 else None
    arg = rng(shape, dtype)
    # Use an arbitrary finite fill_value, since NaNs won't work in a numerical
    # gradient test.
    fun = lambda x: jnp.asarray(x).at[indexer].get(mode=mode, fill_value=7)**2
    check_grads(fun, (arg,), 2, tol, tol, tol)

  def _ReplaceSlicesWithTuples(self, idx):
    """Helper method to replace slices with tuples for dynamic indexing args."""
    if isinstance(idx, slice):
      triple = idx.start, idx.stop, idx.step
      isnone = [i for i, elt in enumerate(triple) if elt is None]
      zeros = itertools.repeat(0)
      nones = itertools.repeat(None)
      out = util.subvals(triple, zip(isnone, zeros))
      return out, lambda out: slice(*util.subvals(out, zip(isnone, nones)))
    elif isinstance(idx, (tuple, list)) and idx:
      t = type(idx)
      elts, packs = zip(*map(self._ReplaceSlicesWithTuples, idx))
      return elts, lambda elts: t((pack(i) for pack, i in zip(packs, elts)))
    else:
      return idx, lambda x: x

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer)
      for name, index_specs in [
        ("OneSliceIndex",
          [IndexSpec(shape=(5,), indexer=slice(1, 3)),
          IndexSpec(shape=(5, 4), indexer=slice(1, 3))]),
        ("TwoSliceIndices",
          [IndexSpec(shape=(5, 4), indexer=(slice(1, 3), slice(0, 2))),
          IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, 2)))]),
        ("NonUnitStrides", [
            IndexSpec(shape=(3,), indexer=slice(None, None, -1)),
            IndexSpec(shape=(3, 3), indexer=slice(0, 3, -2)),
            IndexSpec(shape=(3, 4, 5), indexer=slice(0, 4, 2))
        ]),
        ("OnlyStartOrStopDynamic", [
            IndexSpec(shape=(5, 4), indexer=(slice(None, 3), slice(0, 2))),
            IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, None)))
        ]),
      ]
      for shape, indexer, _ in index_specs
    ],
    dtype=all_dtypes,
  )
  def testDynamicIndexingWithSlicesErrors(self, name, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    @jax.jit
    def fun(x, unpacked_indexer):
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self.assertRaises(IndexError, lambda: fun(*args_maker()))

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer)
      for name, index_specs in [
          ("OneIntIndex",
           [IndexSpec(shape=(3,), indexer=1),
            IndexSpec(shape=(3, 3), indexer=0),
            IndexSpec(shape=(3, 4, 5), indexer=2),
            IndexSpec(shape=(3,), indexer=-1),
            IndexSpec(shape=(3,), indexer=-2)]),
          ("TwoIntIndices",
           [IndexSpec(shape=(3, 3), indexer=(2, 1)),
            IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
            IndexSpec(shape=(3, 4, 5), indexer=(-1, 2))]),
          ("ThreeIntIndices",
           [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
      ]
      for shape, indexer, _ in index_specs
    ],
    dtype=all_dtypes,
  )
  def testDynamicIndexingWithIntegers(self, name, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    def np_fun(x, unpacked_indexer):
      indexer = pack_indexer(unpacked_indexer)
      return np.asarray(x)[indexer]

    def jnp_fun(x, unpacked_indexer):
      indexer = pack_indexer(unpacked_indexer)
      return jnp.array(x)[indexer]

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer)
      for name, index_specs in [
          ("OneIntIndex",
           [IndexSpec(shape=(3,), indexer=1),
            IndexSpec(shape=(3, 3), indexer=0),
            IndexSpec(shape=(3, 4, 5), indexer=2),
            IndexSpec(shape=(3,), indexer=-1),
            IndexSpec(shape=(3,), indexer=-2),
            ]),
          ("TwoIntIndices",
           [IndexSpec(shape=(3, 3), indexer=(2, 1)),
            IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
            IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
            ]),
          ("ThreeIntIndices",
           [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
      ]
      for shape, indexer, _ in index_specs
    ],
    dtype=float_dtypes,
  )
  def testDynamicIndexingWithIntegersGrads(self, name, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    tol = 1e-2 if jnp.finfo(dtype).bits == 32 else None
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    @jax.jit
    def fun(unpacked_indexer, x):
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    arr = rng(shape, dtype)
    check_grads(partial(fun, unpacked_indexer), (arr,), 2, tol, tol, tol)

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer)
      for name, index_specs in ADVANCED_INDEXING_TESTS
      for shape, indexer, _ in index_specs
    ],
    dtype=all_dtypes,
  )
  def testAdvancedIntegerIndexing(self, name, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), indexer]
    np_fun = lambda x, idx: np.asarray(x)[idx]
    jnp_fun = lambda x, idx: jnp.asarray(x)[idx]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(dtype=jtu.dtypes.unsigned + jtu.dtypes.integer)
  def testIndicesNormalizationByType(self, dtype):
    x = jnp.arange(10)
    jaxpr = jax.make_jaxpr(x.__getitem__)(jnp.arange(3, dtype=dtype))
    primitives = [eqn.primitive for eqn in jaxpr.eqns]
    if np.issubdtype(dtype, np.unsignedinteger):
      # Unsigned integers should not require lt, add, and select.
      self.assertEqual(primitives, [lax.convert_element_type_p, lax.broadcast_in_dim_p, lax.gather_p])
    else:
      # May or may not contain convert_element_type.
      self.assertIn(len(primitives), [5, 6])
      self.assertEqual(primitives[:3], [lax.lt_p, lax.add_p, lax.select_n_p])
      self.assertEqual(primitives[-2:], [lax.broadcast_in_dim_p, lax.gather_p])

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer)
      for name, index_specs in [
          ("One1DIntArrayIndex",
           [IndexSpec(shape=(3,), indexer=np.array([0, 1])),
            IndexSpec(shape=(3, 3), indexer=np.array([1, 2, 1])),
            IndexSpec(shape=(3, 4, 5), indexer=np.array([0, 2, 0, 1])),
            IndexSpec(shape=(3,), indexer=np.array([-1, 1])),
            IndexSpec(shape=(3,), indexer=np.array([-2, -1])),
            ]),
          ("One2DIntArrayIndex",
           [IndexSpec(shape=(3,), indexer=np.array([[0, 0]])),
            IndexSpec(shape=(3, 3), indexer=np.array([[1, 2, 1],
                                                       [0, 1, -1]])),
            IndexSpec(shape=(3, 4, 5), indexer=np.array([[0, 2, 0, 1],
                                                          [-1, -2, 1, 0]])),
            ]),
          ("Two1DIntArrayIndicesNoBroadcasting",
           [IndexSpec(shape=(3, 3), indexer=(np.array([0, 1]),
                                             np.array([1, 2]))),
            IndexSpec(shape=(3, 4, 5), indexer=(np.array([0, 2, 0, 1]),
                                                np.array([-1, 0, -1, 2]))),
            ]),
          ("Two1DIntArrayIndicesWithBroadcasting",
           [IndexSpec(shape=(3, 3), indexer=(np.array([[0, 1]]),
                                             np.array([1, 2]))),
            IndexSpec(shape=(3, 4, 5), indexer=(np.array([[0, 2, 0, 1]]),
                                                np.array([-1, 0, -1, 2]))),
            ]),
          ("TupleOfPythonIntsAndIntArrays",
           [IndexSpec(shape=(3, 4, 5), indexer=(0, np.array([0, 1]))),
            IndexSpec(shape=(3, 4, 5), indexer=(0, 1,
                                                np.array([[2, 3, 0, 3]]))),
            ]),
          ("TupleOfListsOfPythonIntsAndIntArrays",
           [IndexSpec(shape=(3, 4, 5), indexer=([0, 1], np.array([0]))),
            IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]],
                                                np.array([[2, 3, 0, 3]]))),
            ]),
      ]
      for shape, indexer, _ in index_specs
    ],
    dtype=float_dtypes,
  )
  def testAdvancedIntegerIndexingGrads(self, name, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    tol = 1e-2 if jnp.finfo(dtype).bits == 32 else None
    arg = rng(shape, dtype)
    fun = lambda x: jnp.asarray(x)[indexer]
    check_grads(fun, (arg,), 2, tol, tol, eps=1.)

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer)
      for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS
      for shape, indexer, _ in index_specs
    ],
    dtype=all_dtypes,
  )
  def testMixedAdvancedIntegerIndexing(self, name, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    indexer_with_dummies = [e if isinstance(e, np.ndarray) else ()
                            for e in indexer]
    substitutes = [(i, e) for i, e in enumerate(indexer)
                   if not isinstance(e, np.ndarray)]
    args_maker = lambda: [rng(shape, dtype), indexer_with_dummies]

    def jnp_fun(x, indexer_with_dummies):
      idx = type(indexer)(util.subvals(indexer_with_dummies, substitutes))
      return jnp.asarray(x)[idx]

    def np_fun(x, indexer_with_dummies):
      idx = type(indexer)(util.subvals(indexer_with_dummies, substitutes))
      return np.asarray(x)[idx]

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testAdvancedIndexingManually(self):
    x = self.rng().randn(3, 4, 5)
    index_array = np.array([0, 2, -1, 0])

    op = lambda x, index_array: x[..., index_array, :]
    cop = jax.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2)

    op = lambda x, index_array: x[..., index_array, :, index_array, None]
    cop = jax.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2)

    op = lambda x, index_array: x[index_array, ..., index_array[:, None], None]
    cop = jax.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2)

  def testUnpacking(self):

    def foo(x):
      a, b, c = x
      return a + b + c

    cfoo = jax.jit(foo)

    a1 = foo(np.arange(3))
    a2 = cfoo(np.arange(3))

    self.assertAllClose(a1, a2)

  def testBooleanIndexingArray1D(self):
    idx = np.array([True, True, False])
    x = jax.device_put(np.arange(3))
    ans = x[idx]
    expected = np.arange(3)[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingList1D(self):
    idx = [True, True, False]
    x = jax.device_put(np.arange(3))
    with self.assertRaisesRegex(TypeError, ARRAY_MSG):
      x[idx]

  def testBooleanIndexingArray2DBroadcast(self):
    idx = np.array([True, True, False, True])
    x = np.arange(8).reshape(4, 2)
    ans = jax.device_put(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingList2DBroadcast(self):
    idx = [True, True, False, True]
    x = np.arange(8).reshape(4, 2)
    with self.assertRaisesRegex(TypeError, ARRAY_MSG):
      jax.device_put(x)[idx]

  def testBooleanIndexingArray2D(self):
    idx = np.array([[True, False],
                     [False, True],
                     [False, False],
                     [True, True]])
    x = np.arange(8).reshape(4, 2)
    ans = jax.device_put(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBoolean1DIndexingWithEllipsis(self):
    # Regression test for https://github.com/google/jax/issues/8412
    x = np.arange(24).reshape(4, 3, 2)
    idx = (..., np.array([True, False]))
    ans = jnp.array(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBoolean1DIndexingWithEllipsis2(self):
    # Regression test for https://github.com/google/jax/issues/9050
    x = np.arange(3)
    idx = (..., np.array([True, False, True]))
    ans = jnp.array(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBoolean1DIndexingWithEllipsis3(self):
    x = np.arange(6).reshape(2, 3)
    idx = (0, ..., np.array([True, False, True]))
    ans = jnp.array(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBoolean2DIndexingWithEllipsis(self):
    x = np.arange(24).reshape(4, 3, 2)
    idx = (..., np.array([[True, False], [True, False], [False, False]]))
    ans = jnp.array(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBoolean1DIndexingWithTrailingEllipsis(self):
    x = np.arange(24).reshape(4, 3, 2)
    idx = (np.array([True, False, True, False]), ...)
    ans = jnp.array(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingDynamicShapeError(self):
    x = np.zeros(3)
    i = np.array([True, True, False])
    self.assertRaises(IndexError, lambda: jax.jit(lambda x, i: x[i])(x, i))

  def testIssue187(self):
    x = jnp.ones((5, 5))
    x[[0, 2, 4], [0, 2, 4]]  # doesn't crash

    x = np.arange(25).reshape((5, 5))
    ans = jax.jit(lambda x: x[[0, 2, 4], [0, 2, 4]])(x)
    expected = x[[0, 2, 4], [0, 2, 4]]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testJVPOfGradOfIndexing(self):
    # Should return a value, even though we didn't pass a symbolic zero as the
    # index tangent.
    x = jnp.ones((3, 4), jnp.float32)
    i = jnp.ones((3,), jnp.int32)
    f = lambda x, i: jnp.sum(x[i])
    primals, tangents = jax.jvp(jax.grad(f), (x, i),
                                (x, np.zeros(i.shape, dtypes.float0)))
    expected = np.broadcast_to(
      np.array([0, 3, 0], dtype=np.float32)[:, None], (3, 4))
    self.assertAllClose(expected, primals)
    self.assertAllClose(np.zeros_like(x), tangents)

  def testSimpleIndexingUsesSlice(self):
    jaxpr = jax.make_jaxpr(lambda x: x[:2, :2])(jnp.ones((3, 4)))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 1)
    self.assertEqual(jaxpr.jaxpr.eqns[-1].primitive, lax.slice_p)

    jaxpr = jax.make_jaxpr(lambda x: x[0, :2, 1])(jnp.ones((3, 4, 5)))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 2)
    self.assertEqual(jaxpr.jaxpr.eqns[-2].primitive, lax.slice_p)
    self.assertEqual(jaxpr.jaxpr.eqns[-1].primitive, lax.squeeze_p)

    jaxpr = jax.make_jaxpr(lambda x: x[0, 0])(jnp.ones((3, 4, 5)))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 2)
    self.assertEqual(jaxpr.jaxpr.eqns[-2].primitive, lax.slice_p)
    self.assertEqual(jaxpr.jaxpr.eqns[-1].primitive, lax.squeeze_p)

    jaxpr = jax.make_jaxpr(lambda x: x[:, 1])(jnp.ones((3, 4, 5)))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 2)
    self.assertEqual(jaxpr.jaxpr.eqns[-2].primitive, lax.slice_p)
    self.assertEqual(jaxpr.jaxpr.eqns[-1].primitive, lax.squeeze_p)

    # Indexing with `Ellipsis` is not lowered to `gather`.
    jaxpr = jax.make_jaxpr(lambda x: x[..., 0])(jnp.ones((3, 4, 5)))
    self.assertLen((jaxpr.jaxpr.eqns), 2)
    self.assertEqual(jaxpr.jaxpr.eqns[-2].primitive, lax.slice_p)
    self.assertEqual(jaxpr.jaxpr.eqns[-1].primitive, lax.squeeze_p)

    # Simple reverses lower to lax.rev_p
    jaxpr = jax.make_jaxpr(lambda x: x[:, ::-1])(jnp.ones((3, 4)))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 1)
    self.assertEqual(jaxpr.jaxpr.eqns[0].primitive, lax.rev_p)

    # Non-static indices produce a dynamic slice
    jaxpr = jax.make_jaxpr(lambda x, i: x[i])(jnp.ones((4,)), 2)
    self.assertEqual(len(jaxpr.jaxpr.eqns), 6)
    self.assertEqual(jaxpr.jaxpr.eqns[-2].primitive, lax.dynamic_slice_p)
    self.assertEqual(jaxpr.jaxpr.eqns[-1].primitive, lax.squeeze_p)

  def testTrivialGatherIsntGenerated(self):
    # https://github.com/google/jax/issues/1621
    jaxpr = jax.make_jaxpr(lambda x: x[:, None])(np.arange(4))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 1)
    self.assertNotIn('gather', str(jaxpr))

    jaxpr = jax.make_jaxpr(lambda x: x[0:6:1])(np.arange(4))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 0)

    jaxpr = jax.make_jaxpr(lambda x: x[:4])(np.arange(4))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 0)

    jaxpr = jax.make_jaxpr(lambda x: x[::-1])(np.arange(4))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 1)
    self.assertEqual(jaxpr.jaxpr.eqns[0].primitive, lax.rev_p)

  def testOOBEmptySlice(self):
    x = jnp.arange(4, dtype='float32')
    self.assertArraysEqual(x[1:0], jnp.empty(0, dtype='float32'))
    self.assertArraysEqual(x[-2:-10], jnp.empty(0, dtype='float32'))
    self.assertArraysEqual(x[5:10], jnp.empty(0, dtype='float32'))

    x = jnp.arange(6, dtype='float32').reshape(2, 3)
    self.assertArraysEqual(x[-1:-4], jnp.empty((0, 3), dtype='float32'))
    self.assertArraysEqual(x[:, 3:2], jnp.empty((2, 0), dtype='float32'))

  def testIndexingEmptyDimension(self):
    # Issue 2671: XLA error when indexing into dimension of size 0
    x = jnp.ones((2, 0))
    # The following work, even on axis 1 of size 0
    with jax.numpy_rank_promotion('allow'):
      _ = x[0, :] + x[0, None] + x[0, 1:] + x[0, 1:3:2]

    with self.assertRaisesRegex(IndexError,
                                "index .* is out of bounds for axis .* with size 0"):
      _ = np.ones((2, 0))[0, 0]  # The numpy error
    with self.assertRaisesRegex(IndexError,
                                "index is out of bounds for axis .* with size 0"):
      _ = x[0, 0]  # JAX indexing
    with self.assertRaisesRegex(IndexError,
                                "index is out of bounds for axis .* with size 0"):
      jax.jit(lambda i: x[0, i])(0)  # JAX indexing under jit

  def testBooleanIndexingWithEmptyResult(self):
    # based on a TensorFlow Probability test that started failing after #1622
    x = jnp.array([-1])
    mask = jnp.array([False])
    ans = x[mask]  # doesn't crash

    expected =  np.array([-1])[np.array([False])]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingShapeMismatch(self):
    # Regression test for https://github.com/google/jax/issues/7329
    x = jnp.arange(4)
    idx = jnp.array([True, False])
    with self.assertRaisesRegex(IndexError, "boolean index did not match shape.*"):
      x[idx]

  def testBooleanIndexingWithNone(self):
    # Regression test for https://github.com/google/jax/issues/18542
    x = jnp.arange(6).reshape(2, 3)
    idx = (None, jnp.array([True, False]))
    ans = x[idx]
    expected = jnp.arange(3).reshape(1, 1, 3)
    self.assertAllClose(ans, expected)

  def testBooleanIndexingWithNoneAndEllipsis(self):
    # Regression test for https://github.com/google/jax/issues/18542
    x = jnp.arange(6).reshape(2, 3)
    mask = jnp.array([True, False, False])
    ans = x[None, ..., mask]
    expected = jnp.array([0, 3]).reshape(1, 2, 1)
    self.assertAllClose(ans, expected)

  def testBooleanIndexingWithEllipsisAndNone(self):
    # Regression test for https://github.com/google/jax/issues/18542
    x = jnp.arange(6).reshape(2, 3)
    mask = jnp.array([True, False, False])
    ans = x[..., None, mask]
    expected = jnp.array([0, 3]).reshape(2, 1, 1)
    self.assertAllClose(ans, expected)

  def testNontrivialBooleanIndexing(self):
    # Test nontrivial corner case in boolean indexing shape validation
    rng = jtu.rand_default(self.rng())
    index = (rng((2, 3), np.bool_), rng((6,), np.bool_))

    args_maker = lambda: [rng((2, 3, 6), np.int32)]
    np_fun = lambda x: np.asarray(x)[index]
    jnp_fun = lambda x: jnp.asarray(x)[index]

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      shape=[(2, 3, 4, 5)],
      idx=[
        np.index_exp[True],
        np.index_exp[False],
        np.index_exp[..., True],
        np.index_exp[..., False],
        np.index_exp[0, :2, True],
        np.index_exp[0, :2, False],
        np.index_exp[:2, 0, True],
        np.index_exp[:2, 0, False],
        np.index_exp[:2, np.array([0, 2]), True],
        np.index_exp[np.array([1, 0]), :, True],
        np.index_exp[True, :, True, :, np.array(True)],
      ]
  )
  def testScalarBooleanIndexing(self, shape, idx):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, np.int32)]
    np_fun = lambda x: np.asarray(x)[idx]
    jnp_fun = lambda x: jnp.asarray(x)[idx]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
      shape=[(2, 3, 4, 5)],
      update_ndim=[0, 1, 2],
      idx=[
        np.index_exp[True],
        np.index_exp[False],
        np.index_exp[..., True],
        np.index_exp[..., False],
        np.index_exp[0, :2, True],
        np.index_exp[0, :2, False],
        np.index_exp[:2, 0, True],
        np.index_exp[:2, 0, False],
        np.index_exp[:2, np.array([0, 2]), True],
        np.index_exp[np.array([1, 0]), :, True],
        np.index_exp[True, :, True, :, np.array(True)],
      ]
  )
  def testScalarBoolUpdate(self, shape, idx, update_ndim):
    update_shape = np.zeros(shape)[idx].shape[-update_ndim:]
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, np.int32), rng(update_shape, np.int32)]
    def np_fun(x, update):
      x = np.array(x, copy=True)
      x[idx] = update
      return x
    jnp_fun = lambda x, update: jnp.asarray(x).at[idx].set(update)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  def testFloatIndexingError(self):
    BAD_INDEX_TYPE_ERROR = "Indexer must have integer or boolean type, got indexer with type"
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jnp.zeros(2)[0.]
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jnp.zeros((2, 2))[(0, 0.)]
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jnp.zeros((2, 2))[(0, 0.)]
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jax.jit(lambda idx: jnp.zeros((2, 2))[idx])((0, 0.))
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jnp.zeros(2).at[0.].add(1.)
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jnp.zeros(2).at[0.].set(1.)

  def testStrIndexingError(self):
    msg = "JAX does not support string indexing"
    with self.assertRaisesRegex(TypeError, msg):
      jnp.zeros(2)['abc']
    with self.assertRaisesRegex(TypeError, msg):
      jnp.zeros((2, 3))[:, 'abc']

  def testIndexOutOfBounds(self):  # https://github.com/google/jax/issues/2245
    x = jnp.arange(5, dtype=jnp.int32) + 1
    self.assertAllClose(x, x[:10])

    idx = jnp.array([-10, -6, -5, -4, 0, 3, 4, 5, 6, 100])
    self.assertArraysEqual(
      x.at[idx].get(mode="clip"),
      jnp.array([1, 1, 1, 2, 1, 4, 5, 5, 5, 5], jnp.int32))
    nan = np.nan
    self.assertArraysEqual(
      x.astype(jnp.float32).at[idx].get(mode="fill"),
      jnp.array([nan, nan, 1, 2, 1, 4, 5, nan, nan, nan], jnp.float32))
    imin = np.iinfo(np.int32).min
    self.assertArraysEqual(
      x.at[idx].get(mode="fill"),
      jnp.array([imin, imin, 1, 2, 1, 4, 5, imin, imin, imin], jnp.int32))
    umax = np.iinfo(np.uint32).max
    self.assertArraysEqual(
      x.astype(np.uint32).at[idx].get(mode="fill"),
      jnp.array([umax, umax, 1, 2, 1, 4, 5, umax, umax, umax], jnp.uint32))
    self.assertArraysEqual(
      x.at[idx].get(mode="fill", fill_value=7),
      jnp.array([7, 7, 1, 2, 1, 4, 5, 7, 7, 7], jnp.int32))

  def testIndexingWeakTypes(self):
    x = lax_internal._convert_element_type(jnp.arange(5), float, weak_type=True)

    a = x.at[0].set(1.0)
    self.assertEqual(a.dtype, x.dtype)
    self.assertTrue(dtypes.is_weakly_typed(a))

    b = x.at[0].add(1.0)
    self.assertEqual(b.dtype, x.dtype)
    self.assertTrue(dtypes.is_weakly_typed(b))

    c = x.at[0].mul(1.0)
    self.assertEqual(c.dtype, x.dtype)
    self.assertTrue(dtypes.is_weakly_typed(c))

  def testIndexingTypePromotion(self):
    def _check(x_type, y_type):
      x = jnp.arange(5, dtype=x_type)
      y = y_type(0)
      out = x.at[0].set(y)
      self.assertEqual(x.dtype, out.dtype)

    @jtu.ignore_warning(category=NumpyComplexWarning,
                        message="Casting complex values to real")
    def _check_warns(x_type, y_type, msg):
      with self.assertWarnsRegex(FutureWarning, msg):
        _check(x_type, y_type)

    def _check_raises(x_type, y_type, msg):
      with self.assertRaisesRegex(ValueError, msg):
        _check(x_type, y_type)

    # Matching dtypes are always OK
    _check(jnp.int32, jnp.int32)
    _check(jnp.float32, jnp.float32)
    _check(jnp.complex64, jnp.complex64)

    # Weakly-typed y values promote.
    _check(jnp.int32, int)
    _check(jnp.float32, int)
    _check(jnp.float32, float)
    _check(jnp.complex64, int)
    _check(jnp.complex64, float)
    _check(jnp.complex64, complex)

    # in standard promotion mode, strong types can promote.
    msg = "scatter inputs have incompatible types"
    with jax.numpy_dtype_promotion('standard'):
      _check(jnp.int32, jnp.int16)
      _check(jnp.float32, jnp.float16)
      _check(jnp.float32, jnp.int32)
      _check(jnp.complex64, jnp.int32)
      _check(jnp.complex64, jnp.float32)

      # TODO(jakevdp): make these _check_raises
      _check_warns(jnp.int16, jnp.int32, msg)
      _check_warns(jnp.int32, jnp.float32, msg)
      _check_warns(jnp.int32, jnp.complex64, msg)
      _check_warns(jnp.float16, jnp.float32, msg)
      _check_warns(jnp.float32, jnp.complex64, msg)

    # in strict promotion mode, strong types do not promote.
    msg = "Input dtypes .* have no available implicit dtype promotion path"
    with jax.numpy_dtype_promotion('strict'):
      _check_raises(jnp.int32, jnp.int16, msg)
      _check_raises(jnp.float32, jnp.float16, msg)
      _check_raises(jnp.float32, jnp.int32, msg)
      _check_raises(jnp.complex64, jnp.int32, msg)
      _check_raises(jnp.complex64, jnp.float32, msg)

      _check_raises(jnp.int16, jnp.int32, msg)
      _check_raises(jnp.int32, jnp.float32, msg)
      _check_raises(jnp.int32, jnp.complex64, msg)
      _check_raises(jnp.float16, jnp.float32, msg)
      _check_raises(jnp.float32, jnp.complex64, msg)

  def testWrongNumberOfIndices(self):
    with self.assertRaisesRegex(
        IndexError,
        "Too many indices for array: 2 non-None/Ellipsis indices for dim 1."):
      jnp.zeros(3)[:, 5]


def _broadcastable_shapes(shape):
  """Returns all shapes that broadcast to `shape`."""
  def f(rshape):
    yield []
    if rshape:
      for s in f(rshape[1:]):
        yield rshape[0:1] + s
      if rshape[0] != 1:
        for s in f(rshape[1:]):
          yield [1] + s
  for x in f(list(reversed(shape))):
    yield list(reversed(x))


# TODO(jakevdp): move this implementation to jax.dtypes & use in scatter?
def _can_cast(from_, to):
  with jax.numpy_dtype_promotion('standard'):
    return lax.dtype(to) == dtypes.result_type(from_, to)


def _compatible_dtypes(op, dtype, inexact=False):
  if op == UpdateOps.ADD:
    return [dtype]
  elif inexact:
    return [dt for dt in float_dtypes if _can_cast(dt, dtype)]
  else:
    return [dt for dt in all_dtypes if _can_cast(dt, dtype)]


class UpdateOps(enum.Enum):
  UPDATE = 0
  ADD = 1
  MUL = 2
  DIV = 3
  POW = 4
  MIN = 5
  MAX = 6

  def np_fn(op, indexer, x, y):
    x = x.copy()
    x[indexer] = {
      UpdateOps.UPDATE: lambda: y,
      UpdateOps.ADD: lambda: x[indexer] + y,
      UpdateOps.MUL: lambda: x[indexer] * y,
      UpdateOps.DIV: jtu.ignore_warning(category=RuntimeWarning)(
        lambda: x[indexer] / y.astype(x.dtype)),
      UpdateOps.POW: jtu.ignore_warning(category=RuntimeWarning)(
        lambda: x[indexer] ** y.astype(x.dtype)),
      UpdateOps.MIN: lambda: np.minimum(x[indexer], y),
      UpdateOps.MAX: lambda: np.maximum(x[indexer], y),
    }[op]()
    return x

  def jax_fn(op, indexer, x, y, indices_are_sorted=False,
             unique_indices=False, mode=None):
    x = jnp.array(x)
    return {
      UpdateOps.UPDATE: x.at[indexer].set,
      UpdateOps.ADD: x.at[indexer].add,
      UpdateOps.MUL: x.at[indexer].multiply,
      UpdateOps.DIV: x.at[indexer].divide,
      UpdateOps.POW: x.at[indexer].power,
      UpdateOps.MIN: x.at[indexer].min,
      UpdateOps.MAX: x.at[indexer].max,
    }[op](y, indices_are_sorted=indices_are_sorted,
          unique_indices=unique_indices, mode=mode)

  def dtypes(op):
    if op == UpdateOps.UPDATE:
      return all_dtypes
    elif op == UpdateOps.DIV or op == UpdateOps.POW:
      return jtu.dtypes.inexact
    else:
      return default_dtypes

def _update_tol(op):
  if op == UpdateOps.POW:
    f32_tol = 2e-4 if jtu.test_device_matches(["tpu"]) else 1e-5
    tol = {np.float32: f32_tol, np.complex64: f32_tol, np.complex128: 1e-14}
  else:
    tol = {np.complex128: 1e-14}
  return tol


class IndexedUpdateTest(jtu.JaxTestCase):

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer, update_shape=update_shape)
     for name, index_specs in STATIC_INDEXING_TESTS
     for shape, indexer, index_shape in index_specs
     for update_shape in _broadcastable_shapes(index_shape)
    ],
    [dict(op=op, dtype=dtype, update_dtype=update_dtype)
     for op in UpdateOps
     for dtype in UpdateOps.dtypes(op)
     for update_dtype in _compatible_dtypes(op, dtype)
    ],
    mode=MODES,
  )
  def testStaticIndexing(self, name, shape, dtype, update_shape, update_dtype,
                         indexer, op, mode):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y, mode=mode)
    with jtu.strict_promotion_if_dtypes_match([dtype, update_dtype]):
      self._CheckAgainstNumpy(np_fn, jax_fn, args_maker, tol=_update_tol(op))
      self._CompileAndCheck(jax_fn, args_maker)

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer, update_shape=update_shape)
     for name, index_specs in ADVANCED_INDEXING_TESTS_NO_REPEATS
     for shape, indexer, index_shape in index_specs
     for update_shape in _broadcastable_shapes(index_shape)
    ],
    [dict(op=op, dtype=dtype, update_dtype=update_dtype)
     for op in UpdateOps
     for dtype in UpdateOps.dtypes(op)
     for update_dtype in _compatible_dtypes(op, dtype)
    ],
  )
  def testAdvancedIndexing(self, name, shape, dtype, update_shape, update_dtype,
                           indexer, op):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y,
                                           unique_indices=True)
    with jtu.strict_promotion_if_dtypes_match([dtype, update_dtype]):
      self._CheckAgainstNumpy(np_fn, jax_fn, args_maker, tol=_update_tol(op))
      self._CompileAndCheck(jax_fn, args_maker)

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer, update_shape=update_shape)
     for name, index_specs in ADVANCED_INDEXING_TESTS_NO_REPEATS_SORTED
     for shape, indexer, index_shape in index_specs
     for update_shape in _broadcastable_shapes(index_shape)
    ],
    [dict(op=op, dtype=dtype, update_dtype=update_dtype)
     for op in UpdateOps
     for dtype in UpdateOps.dtypes(op)
     for update_dtype in _compatible_dtypes(op, dtype)
    ],
  )
  def testAdvancedIndexingSorted(self, name, shape, dtype, update_shape,
                                 update_dtype, indexer, op):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    jax_fn = lambda x, y: UpdateOps.jax_fn(
      op, indexer, x, y, indices_are_sorted=True, unique_indices=True)
    with jtu.strict_promotion_if_dtypes_match([dtype, update_dtype]):
      self._CheckAgainstNumpy(np_fn, jax_fn, args_maker, check_dtypes=True,
                              tol=_update_tol(op))
      self._CompileAndCheck(jax_fn, args_maker, check_dtypes=True)

  @jtu.sample_product(
    [dict(name=name, shape=shape, indexer=indexer, update_shape=update_shape)
     for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS
     for shape, indexer, index_shape in index_specs
     for update_shape in _broadcastable_shapes(index_shape)
    ],
    [dict(op=op, dtype=dtype, update_dtype=update_dtype)
     for op in UpdateOps
     for dtype in UpdateOps.dtypes(op)
     for update_dtype in _compatible_dtypes(op, dtype)
    ],
  )
  def testMixedAdvancedIndexing(self, name, shape, dtype, update_shape,
                                update_dtype, indexer, op):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y)
    with jtu.strict_promotion_if_dtypes_match([dtype, update_dtype]):
      self._CheckAgainstNumpy(np_fn, jax_fn, args_maker, tol=_update_tol(op))
      self._CompileAndCheck(jax_fn, args_maker)

  @jtu.sample_product(
    [dict(name=name, mode=mode, shape=shape, indexer=indexer,
          update_shape=update_shape)
     for mode in [None] + MODES
     for name, index_specs in (
       STATIC_INDEXING_TESTS if mode == "promise_in_bounds" else
       STATIC_INDEXING_TESTS + STATIC_INDEXING_OUT_OF_BOUNDS_TESTS)
     for shape, indexer, index_shape in index_specs
     for update_shape in _broadcastable_shapes(index_shape)
    ],
    [dict(op=op, dtype=dtype, update_dtype=update_dtype)
     for op in [UpdateOps.ADD, UpdateOps.MUL, UpdateOps.UPDATE]
     for dtype in float_dtypes
     for update_dtype in _compatible_dtypes(op, dtype, inexact=True)
    ],
  )
  def testStaticIndexingGrads(self, name, shape, dtype, update_shape,
                              update_dtype, indexer, op, mode):
    rng = jtu.rand_default(self.rng())
    jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y, mode=mode,
    unique_indices=True)
    x = rng(shape, dtype)
    y = rng(update_shape, update_dtype)
    with jtu.strict_promotion_if_dtypes_match([dtype, update_dtype]):
      check_grads(jax_fn, (x, y), 2, rtol=1e-3, atol=1e-3, eps=1.)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=name, unique_indices=unique_indices, shape=shape,
            indexer=indexer, update_shape=update_shape)
      for name, index_specs in (
        ADVANCED_INDEXING_TESTS_NO_REPEATS if unique_indices
        else ADVANCED_INDEXING_TESTS)
      for shape, indexer, index_shape in index_specs
      for update_shape in _broadcastable_shapes(index_shape)
      ],
      [dict(op=op, dtype=dtype, update_dtype=update_dtype)
      for op in (
        [UpdateOps.ADD, UpdateOps.MUL, UpdateOps.UPDATE] if unique_indices
        else [UpdateOps.ADD])
      for dtype in float_dtypes
      for update_dtype in _compatible_dtypes(op, dtype, inexact=True)
      ],
    )
    for unique_indices in [False, True]
  ))
  def testAdvancedIndexingGrads(self, name, shape, dtype, update_shape,
                                update_dtype, indexer, op, unique_indices):
    rng = jtu.rand_default(self.rng())
    jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y,
                                           unique_indices=unique_indices)
    x = rng(shape, dtype)
    y = rng(update_shape, update_dtype)
    with jtu.strict_promotion_if_dtypes_match([dtype, update_dtype]):
      check_grads(jax_fn, (x, y), 2, rtol=1e-3, atol=1e-3, eps=1.)

  def testIndexMulGradFailsIfNotUnique(self):
    y = jnp.ones((10,), jnp.int32)
    f = lambda x, z: x.at[y].mul(z)

    x = jnp.ones((100,), jnp.float32)
    z = jnp.ones((10,), jnp.float32)
    with self.assertRaises(NotImplementedError,
                           msg="scatter_mul gradients are only implemented if "
                           "`unique_indices=True`"):
      jax.jvp(f, (x, z), (x, z))

  def testSegmentSumBehavior(self):
    # testAdvancedIndexing compares against NumPy, and as a result doesn't check
    # repeated indices. This test is just a simple manual check, based on
    # https://www.tensorflow.org/api_docs/python/tf/math/segment_sum
    data = np.array([5, 1, 7, 2, 3, 4, 1, 3], dtype=float)
    segment_ids = np.array([0, 0, 0, 1, 2, 2, 3, 3])

    ans = jnp.zeros_like(data, shape=np.max(segment_ids) + 1).at[segment_ids].add(data)
    expected = np.array([13, 2, 7, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSegmentSum(self):
    data = jnp.array([5, 1, 7, 2, 3, 4, 1, 3])
    segment_ids = jnp.array([0, 0, 0, 1, 2, 2, 3, 3])

    # test with explicit num_segments
    ans = ops.segment_sum(data, segment_ids, num_segments=4)
    expected = jnp.array([13, 2, 7, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # test with explicit num_segments larger than the higher index.
    ans = ops.segment_sum(data, segment_ids, num_segments=5)
    expected = jnp.array([13, 2, 7, 4, 0])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # test without explicit num_segments
    ans = ops.segment_sum(data, segment_ids)
    expected = jnp.array([13, 2, 7, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # test with negative segment ids and segment ids larger than num_segments,
    # that will be wrapped with the `mod`.
    segment_ids = jnp.array([0, 4, 8, 1, 2, -6, -1, 3])
    ans = ops.segment_sum(data, segment_ids, num_segments=4)
    expected = jnp.array([5, 2, 3, 3])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # test with negative segment ids and without explicit num_segments
    # such as num_segments is defined by the smaller index.
    segment_ids = jnp.array([3, 3, 3, 4, 5, 5, -7, -6])
    ans = ops.segment_sum(data, segment_ids)
    expected = jnp.array([0, 0, 0, 13, 2, 7])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSegmentSumOutOfBounds(self):
    def fn(data, segment_ids):
      return jax.ops.segment_sum(data, segment_ids, num_segments).sum()

    data = np.array([0, 0], dtype=np.float32)
    num_segments = 2
    segment_ids = np.array([2, 3])
    val, grad = jax.value_and_grad(fn)(data, segment_ids)
    self.assertAllClose(val, np.array(0., np.float32))
    self.assertAllClose(grad, np.array([0., 0.], np.float32))


  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(reducer=reducer, op=op, identity=identity)],
      dtype=[np.bool_],
      shape=[(8,), (7, 4), (6, 4, 2)],
      bucket_size=[None, 2],
      num_segments=[None, 1, 3],
    )
    for reducer, op, identity in [
      (ops.segment_min, np.minimum, True),
      (ops.segment_max, np.maximum, False),
    ]))
  def testSegmentReduceBoolean(self, shape, dtype, reducer, op, identity,
                               num_segments, bucket_size):
    rng = jtu.rand_default(self.rng())
    idx_rng = jtu.rand_int(self.rng(), low=-2, high=3)
    args_maker = lambda: [rng(shape, dtype), idx_rng(shape[:1], jnp.int32)]

    if np.issubdtype(dtype, np.integer):
      if np.isposinf(identity):
        identity = np.iinfo(dtype).max
      elif np.isneginf(identity):
        identity = np.iinfo(dtype).min

    jnp_fun = lambda data, segment_ids: reducer(
      data, segment_ids, num_segments=num_segments, bucket_size=bucket_size)

    def np_fun(data, segment_ids):
      size = num_segments if num_segments is not None else (segment_ids.max() + 1)
      out = np.full((size,) + shape[1:], identity, dtype)
      for i, val in zip(segment_ids, data):
        if 0 <= i < size:
          out[i] = op(out[i], val).astype(dtype)
      return out

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    if num_segments is not None:
      self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(reducer=reducer, op=op, identity=identity)],
      dtype=default_dtypes,
      shape=[(8,), (7, 4), (6, 4, 2)],
      bucket_size=[None, 2],
      num_segments=[None, 1, 3],
    )
    for reducer, op, identity in [
      (ops.segment_sum, np.add, 0),
      (ops.segment_prod, np.multiply, 1),
      (ops.segment_min, np.minimum, float('inf')),
      (ops.segment_max, np.maximum, -float('inf')),
    ]))
  def testSegmentReduce(self, shape, dtype, reducer, op, identity, num_segments, bucket_size):
    rng = jtu.rand_default(self.rng())
    idx_rng = jtu.rand_int(self.rng(), low=-2, high=3)
    args_maker = lambda: [rng(shape, dtype), idx_rng(shape[:1], jnp.int32)]

    if np.issubdtype(dtype, np.integer):
      if np.isposinf(identity):
        identity = np.iinfo(dtype).max
      elif np.isneginf(identity):
        identity = np.iinfo(dtype).min

    jnp_fun = lambda data, segment_ids: reducer(
      data, segment_ids, num_segments=num_segments, bucket_size=bucket_size)

    def np_fun(data, segment_ids):
      size = num_segments if num_segments is not None else (segment_ids.max() + 1)
      out = np.full((size,) + shape[1:], identity, dtype)
      for i, val in zip(segment_ids, data):
        if 0 <= i < size:
          out[i] = op(out[i], val).astype(dtype)
      return out

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    if num_segments is not None:
      self._CompileAndCheck(jnp_fun, args_maker)

  def testIndexDtypeError(self):
    # https://github.com/google/jax/issues/2795
    jnp.array(1)  # get rid of startup warning
    with self.assertNoWarnings():
      jnp.zeros(5).at[::2].set(1)

  @jtu.sample_product(
    [dict(idx=idx, idx_type=idx_type)
     for idx, idx_type in [
       ([0], "array"),
       ([0, 0], "array"),
       ([[0, 0]], "tuple"),
       ([0, [0, 1]], "tuple"),
       ([0, np.arange(2)], "tuple"),
       ([0, None], "tuple"),
       ([0, slice(None)], "tuple"),
      ]
    ],
  )
  def testIndexSequenceDeprecation(self, idx, idx_type):
    normalize = {"array": np.array, "tuple": tuple}[idx_type]
    msg = {"array": ARRAY_MSG, "tuple": TUPLE_MSG}[idx_type]
    x = jnp.arange(6).reshape(3, 2)

    with self.assertRaisesRegex(TypeError, msg):
      x[idx]
    with self.assertNoWarnings():
      x[normalize(idx)]

    with self.assertRaisesRegex(TypeError, msg):
      x.at[idx].set(0)
    with self.assertNoWarnings():
      x.at[normalize(idx)].set(0)

  def testIndexedUpdateAliasingBug(self):
    # https://github.com/google/jax/issues/7461
    fn = lambda x: x.at[1:].set(1 + x[:-1])
    y = jnp.zeros(8)
    self.assertArraysEqual(fn(y), jax.jit(fn)(y))

  def testScatterValuesCastToTargetDType(self):
    # https://github.com/google/jax/issues/15505
    a = jnp.zeros(1, dtype=jnp.uint32)
    val = 2**32 - 1  # too large for int32

    b = a.at[0].set(jnp.uint32(val))
    self.assertEqual(int(b[0]), val)

    c = a.at[0].set(val)
    self.assertEqual(int(c[0]), val)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
