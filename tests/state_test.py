# Copyright 2022 The JAX Authors.
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

from collections.abc import Callable, Sequence
from functools import partial
import itertools as it
from typing import Any, NamedTuple, Union

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import jax
from jax import random
from jax import lax
from jax._src import core
from jax._src import config
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax._src import test_util as jtu
from jax._src.util import tuple_insert
import jax.numpy as jnp
from jax._src.lax.control_flow import for_loop

try:
  import hypothesis as hp
  import hypothesis.extra.numpy as hnp
  import hypothesis.strategies as hps
  CAN_USE_HYPOTHESIS = True
except (ModuleNotFoundError, ImportError):
  CAN_USE_HYPOTHESIS = False

from jax._src.state.discharge import (run_state, run_state_reference,
                                      discharge_state)
from jax._src.state.primitives import (get_p, swap_p, addupdate_p,
                                       ref_addupdate, ref_get, ref_set,
                                       ref_swap)
from jax._src.state.types import (shaped_array_ref, ReadEffect, WriteEffect,
                                  AccumEffect, AbstractRef)

config.parse_flags_with_absl()
jtu.setup_hypothesis()


class StatePrimitivesTest(jtu.JaxTestCase):

  def test_get_abstract_aval_must_take_in_refs(self):
    ref_aval = core.ShapedArray((), jnp.float32)
    def f(x_ref):
      return [ref_get(x_ref, ())]
    with self.assertRaises(ValueError):
      pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval])

  @parameterized.named_parameters(
      dict(testcase_name="trivial_get", ref_shape=(1, 2),
           ref_dtype=jnp.float32,
           idx=(), out_shape=(1, 2), out_dtype=jnp.float32),
      dict(testcase_name="get_with_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32,
           idx=(0,), out_shape=(2,), out_dtype=jnp.float32),
      dict(testcase_name="get_with_nonleading_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32,
           idx=(slice(None), 0), out_shape=(1,), out_dtype=jnp.float32),
      dict(testcase_name="get_with_array_index", ref_shape=(1, 2, 3, 4),
           ref_dtype=jnp.float32,
           idx=(np.array([0, 1]),), out_shape=(2, 2, 3, 4),
           out_dtype=jnp.float32),
      dict(testcase_name="get_with_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(np.array([0, 1]), np.array([0, 1])),
           out_shape=(2, 2, 4), out_dtype=jnp.float32),
      dict(testcase_name="get_with_nonleading_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(None), np.array([0, 1]), slice(None), np.array([0, 1])),
           out_shape=(2, 1, 2), out_dtype=jnp.float32),
      dict(testcase_name="get_with_nontrivial_slice",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(0, 1), np.array([0, 1]), slice(None), np.array([0, 1])),
           out_shape=(2, 1, 2), out_dtype=jnp.float32),
      dict(testcase_name="get_with_nontrivial_slice2",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(0, 1), slice(1, 3), slice(None), slice(None)),
           out_shape=(1, 2, 2, 4), out_dtype=jnp.float32),
      dict(testcase_name="get_with_ref_simple_at",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(1, 3), slice(None), slice(None)),
           out_shape=(2, 2, 4), out_dtype=jnp.float32,
           at_indices=((0,),)),
      dict(testcase_name="get_with_ref_simple_at2",
           ref_shape=(6, 1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(0, 2), slice(0, 1), slice(1, 3), slice(None), slice(None)),
           out_shape=(2, 1, 2, 2, 4), out_dtype=jnp.float32,
           at_indices=((slice(2, 6),),)),
      dict(testcase_name="get_with_ref_multiple_at",
           ref_shape=(1, 3, 5, 4), ref_dtype=jnp.float32,
           idx=(slice(None), slice(None), slice(0, 2)),
           out_shape=(3, 1, 2), out_dtype=jnp.float32,
           at_indices=((0,), (slice(None), slice(0, 1)))),
  )
  def test_get_abstract_eval(self, ref_shape, ref_dtype, idx, out_shape=None,
                             out_dtype=None, at_indices=(),
                             should_error=False):
    ref_aval = AbstractRef(core.ShapedArray(ref_shape, ref_dtype))
    def f(x_ref):
      for at_idx in at_indices:
        x_ref = x_ref.at[at_idx]
      out = ref_get(x_ref, idx)
      return [out]
    if should_error:
      with self.assertRaises(Exception):
        pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval])
    else:
      jaxpr, out_avals, _, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval])
      self.assertSetEqual(jaxpr.effects,
                          {ReadEffect(len(jaxpr.constvars))})
      self.assertLen(out_avals, 1)
      out_aval, = out_avals
      self.assertIsInstance(out_aval, core.ShapedArray)
      self.assertEqual(out_aval.shape, out_shape)
      self.assertEqual(out_aval.dtype, out_dtype)

  def test_swap_abstract_eval_must_take_in_refs(self):
    ref_aval = core.ShapedArray((), jnp.float32)
    val_aval = core.ShapedArray((), jnp.float32)
    def f(x_ref, val):
      return [ref_swap(x_ref, (), val)]
    with self.assertRaises(ValueError):
      pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval, val_aval])

  @parameterized.named_parameters(
      dict(testcase_name="invalid_val_shape", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(), should_error=True),
      dict(testcase_name="invalid_val_shape_slice", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(slice(None),), should_error=True),
      dict(testcase_name="trivial_swap", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(1, 2), val_dtype=jnp.float32,
           idx=(), out_shape=(1, 2), out_dtype=jnp.float32),
      dict(testcase_name="bad_dtype", ref_shape=(1, 2),
           ref_dtype=jnp.int32, val_shape=(1, 2), val_dtype=jnp.float32,
           idx=(), should_error=True),
      dict(testcase_name="swap_with_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(0,), out_shape=(2,), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_nonleading_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(1,), val_dtype=jnp.float32,
           idx=(slice(None), 0), out_shape=(1,), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_nonleading_index_bad_val", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(slice(None), 0), should_error=True),
      dict(testcase_name="swap_with_array_index", ref_shape=(1, 2, 3, 4),
           ref_dtype=jnp.float32, val_shape=(2, 2, 3, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]),), out_shape=(2, 2, 3, 4),
           out_dtype=jnp.float32),
      dict(testcase_name="swap_with_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 2, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]), np.array([0, 1])),
           out_shape=(2, 2, 4), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_nonleading_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 1, 2), val_dtype=jnp.float32,
           idx=(slice(None), np.array([0, 1]), slice(None), np.array([0, 1])),
           out_shape=(2, 1, 2), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_nontrivial_slice",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(0, 1), np.array([0, 1]), slice(None), np.array([0, 1])),
           val_shape=(2, 1, 2), val_dtype=jnp.float32,
           out_shape=(2, 1, 2), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_nontrivial_slice2",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(0, 1), slice(1, 3), slice(None), slice(None)),
           val_shape=(1, 2, 2, 4), val_dtype=jnp.float32,
           out_shape=(1, 2, 2, 4), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_ref_simple_at",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(0, 1), slice(1, 3), slice(None),),
           val_shape=(1, 1, 4), val_dtype=jnp.float32,
           out_shape=(1, 1, 4), out_dtype=jnp.float32,
           at_indices=((0,),),),
      dict(testcase_name="swap_with_ref_simple_at2",
           ref_shape=(4, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(None), slice(0, 1), slice(1, 3), slice(None),),
           val_shape=(2, 1, 1, 4), val_dtype=jnp.float32,
           out_shape=(2, 1, 1, 4), out_dtype=jnp.float32,
           at_indices=((slice(0, 2),),),),
      dict(testcase_name="swap_with_ref_multiple_at2",
           ref_shape=(1, 4, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(None), slice(0, 1), slice(1, 3), slice(None),),
           val_shape=(2, 1, 1, 4), val_dtype=jnp.float32,
           out_shape=(2, 1, 1, 4), out_dtype=jnp.float32,
           at_indices=((slice(None), slice(0, 2),), (0,)),),
  )
  def test_swap_abstract_eval(self, ref_shape, ref_dtype,
      val_shape, val_dtype, idx, out_shape=None, out_dtype=None,
      at_indices=(), should_error=False):
    ref_aval = AbstractRef(core.ShapedArray(ref_shape, ref_dtype))
    val_aval = core.ShapedArray(val_shape, val_dtype)
    def f(x_ref, val):
      for at_idx in at_indices:
        x_ref = x_ref.at[at_idx]
      out = ref_swap(x_ref, idx, val)
      return [out]
    if should_error:
      with self.assertRaises(Exception):
        pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval, val_aval])
    else:
      jaxpr, out_avals, _, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval, val_aval])
      self.assertSetEqual(jaxpr.effects,
                          {WriteEffect(len(jaxpr.constvars))})
      self.assertLen(out_avals, 1)
      out_aval, = out_avals
      self.assertIsInstance(out_aval, core.ShapedArray)
      self.assertEqual(out_aval.shape, out_shape)
      self.assertEqual(out_aval.dtype, out_dtype)

  @parameterized.named_parameters(
      dict(testcase_name="invalid_val_shape", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(), should_error=True),
      dict(testcase_name="invalid_val_shape_slice", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(slice(None),), should_error=True),
      dict(testcase_name="trivial_addupdate", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(1, 2), val_dtype=jnp.float32,
           idx=(),),
      dict(testcase_name="bad_dtype", ref_shape=(1, 2),
           ref_dtype=jnp.int32, val_shape=(1, 2), val_dtype=jnp.float32,
           idx=(), should_error=True),
      dict(testcase_name="addupdate_with_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(0,),),
      dict(testcase_name="addupdate_with_nonleading_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(1,), val_dtype=jnp.float32,
           idx=(slice(None), 0)),
      dict(testcase_name="addupdate_with_nonleading_index_bad_val", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(slice(None), 0), should_error=True),
      dict(testcase_name="addupdate_with_array_index", ref_shape=(1, 2, 3, 4),
           ref_dtype=jnp.float32, val_shape=(2, 2, 3, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]),)),
      dict(testcase_name="addupdate_with_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 2, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]), np.array([0, 1]))),
      dict(testcase_name="addupdate_with_nonleading_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 1, 2), val_dtype=jnp.float32,
           idx=(slice(None), np.array([0, 1]), slice(None), np.array([0, 1]))),
      dict(testcase_name="ref_with_simple_at",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 2), val_dtype=jnp.float32,
           idx=(np.array([0, 1]), slice(None), np.array([0, 1])),
           at_indices=((0,),)),
      dict(testcase_name="ref_with_simple_at2",
           ref_shape=(3, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 3, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]), slice(None), np.array([0, 1])),
           at_indices=((slice(0, 3),),)),
      dict(testcase_name="ref_with_multiple_at",
           ref_shape=(3, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 2), val_dtype=jnp.float32,
           idx=(np.array([0, 1]), slice(None), np.array([0, 1])),
           at_indices=((slice(0, 3),), (0,))),
      dict(testcase_name="ref_with_multiple_at2",
           ref_shape=(3, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 2), val_dtype=jnp.float32,
           idx=(np.array([0, 1]), slice(None), np.array([0, 1])),
           at_indices=((slice(None), slice(0, 3),), (0,))),
  )
  def test_addupdate_abstract_eval(self, ref_shape, ref_dtype,
      val_shape, val_dtype, idx, at_indices=(), should_error=False):
    ref_aval = AbstractRef(core.ShapedArray(ref_shape, ref_dtype))
    val_aval = core.ShapedArray(val_shape, val_dtype)
    def f(x_ref, val):
      for at_idx in at_indices:
        x_ref = x_ref.at[at_idx]
      ref_addupdate(x_ref, idx, val)
      return []
    if should_error:
      with self.assertRaises(Exception):
        pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval, val_aval])
    else:
      jaxpr, out_avals, _, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval, val_aval])
      self.assertSetEqual(jaxpr.effects,
                          {AccumEffect(len(jaxpr.constvars))})
      self.assertLen(out_avals, 0)

  def test_addupdate_abstract_eval_must_take_in_refs(self):
    ref_aval = core.ShapedArray((), jnp.float32)
    val_aval = core.ShapedArray((), jnp.float32)
    def f(x_ref, val):
      return [ref_addupdate(x_ref, (), val)]
    with self.assertRaises(ValueError):
      pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval, val_aval])

  def test_can_represent_get_and_swap_in_jaxprs(self):

    def body(x):
      x[()] = jnp.int32(1)
      x[()] = jnp.int32(2)
      return (x[()],)
    jaxpr, out_avals, consts, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((), jnp.int32)])
    self.assertLen(consts, 0)
    self.assertListEqual(out_avals, [core.ShapedArray((), jnp.int32)])
    self.assertEqual(jaxpr.eqns[0].primitive, swap_p)
    self.assertEqual(jaxpr.eqns[1].primitive, swap_p)
    self.assertEqual(jaxpr.eqns[2].primitive, get_p)

  def test_can_represent_addupdate_in_jaxprs(self):

    def body(x):
      ref_addupdate(x, (), jnp.int32(1))
      return (x[()],)
    jaxpr, out_avals, consts, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((), jnp.int32)])
    self.assertLen(consts, 0)
    self.assertListEqual(out_avals, [core.ShapedArray((), jnp.int32)])
    self.assertEqual(jaxpr.eqns[0].primitive, addupdate_p)

  def test_get_custom_pretty_printing_rule(self):
    def body(x_ref):
      x = x_ref[()]
      return [x]
    jaxpr, _ , _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((), jnp.int32)])
    self.assertIn("b:i32[] <- a[]", jaxpr.pretty_print(use_color=False))

    def body(x_ref):
      x = x_ref[:, 0]
      return [x]
    jaxpr, _ , _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((1, 2), jnp.int32)])
    self.assertIn("b:i32[1] <- a[:,0]", jaxpr.pretty_print(use_color=False))

  def test_set_custom_pretty_printing_rule(self):
    def body(x_ref):
      x_ref[()] = jnp.int32(2)
      return []
    jaxpr, _ , _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((), jnp.int32)])
    self.assertIn("a[] <- 2", jaxpr.pretty_print(use_color=False))

    def body(x_ref, val):
      x_ref[:, 0] = val
      return []
    jaxpr, _ , _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((1, 2), jnp.int32),
                             core.ShapedArray((1,), jnp.int32)])
    self.assertIn("a[:,0] <- b", jaxpr.pretty_print(use_color=False))

  def test_swap_custom_pretty_printing_rule(self):
    def body(x_ref):
      x = ref_swap(x_ref, (), jnp.int32(2))
      return [x]
    jaxpr, _ , _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((), jnp.int32)])
    self.assertIn("b:i32[], a[] <- a[], 2", jaxpr.pretty_print(use_color=False))

    def body(x_ref, val):
      x = ref_swap(x_ref, (slice(None), 0), val)
      return [x]
    jaxpr, _ , _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((1, 2), jnp.int32),
                             core.ShapedArray((1,), jnp.int32)])
    self.assertIn("c:i32[1], a[:,0] <- a[:,0], b",
                  jaxpr.pretty_print(use_color=False))

  def test_addupdate_custom_pretty_printing_rule(self):
    def body(x_ref):
      ref_addupdate(x_ref, (), jnp.int32(2))
      return []
    jaxpr, _ , _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((), jnp.int32)])

    self.assertIn("a[] += 2", jaxpr.pretty_print(use_color=False))

    def body(x_ref, val):
      ref_addupdate(x_ref, (slice(None), 0), val)
      return []
    jaxpr, _ , _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [shaped_array_ref((1, 2), jnp.int32),
                             core.ShapedArray((1,), jnp.int32)])
    self.assertIn("a[:,0] += b", jaxpr.pretty_print(use_color=False))


  def test_get_jvp(self):

    def f(r):
      x = r[()]
      return jnp.cos(x)

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [shaped_array_ref((), jnp.dtype('float32')),
                shaped_array_ref((), jnp.dtype('float32'))]
    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, get_p)
    self.assertEqual(jaxpr.eqns[1].primitive, get_p)

  def test_swap_jvp(self):

    def f(a):
      x = a[()]
      a[()] = jnp.sin(x)
      return a[()]

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [shaped_array_ref((), jnp.dtype('float32')),
                shaped_array_ref((), jnp.dtype('float32'))]
    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, get_p)
    self.assertEqual(jaxpr.eqns[1].primitive, get_p)
    self.assertEqual(jaxpr.eqns[2].primitive, lax.sin_p)
    self.assertEqual(jaxpr.eqns[3].primitive, lax.cos_p)
    self.assertEqual(jaxpr.eqns[4].primitive, lax.mul_p)
    self.assertEqual(jaxpr.eqns[5].primitive, swap_p)
    self.assertEqual(jaxpr.eqns[6].primitive, swap_p)

  def test_addupdate_jvp(self):

    def f(a):
      ref_addupdate(a, (), jnp.float32(1.))
      return a[()]

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [shaped_array_ref((), jnp.dtype('float32')),
                shaped_array_ref((), jnp.dtype('float32'))]
    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, addupdate_p)
    self.assertEqual(jaxpr.eqns[1].primitive, addupdate_p)
    self.assertEqual(jaxpr.eqns[2].primitive, get_p)
    self.assertEqual(jaxpr.eqns[3].primitive, get_p)

  @jtu.sample_product(
    [dict(ref_shape=ref_shape, ref_bdim=ref_bdim, idx_shape=idx_shape,
          indexed_dims=indexed_dims, idx_bdims=idx_bdims, out_bdim=out_bdim)
     for ref_shape in [(1,), (2, 3), (4, 5, 6)]
     for ref_bdim in range(1 + len(ref_shape))
     for idx_shape in [(), (1,), (2,), (5, 6)]
     for indexed_dims in it.product([True, False], repeat=len(ref_shape))
     for idx_bdims in it.product([None, *range(1 + len(idx_shape))],
                                 repeat=sum(indexed_dims))
     for out_bdim in range(1 + len(ref_shape) - sum(indexed_dims)
                           + len(idx_shape) * any(indexed_dims))
    ],
    op=[
        lambda x_ref, indexer: [x_ref[indexer]],
        lambda x_ref, indexer: [
            ref_swap(x_ref, indexer,
                            jnp.ones(x_ref.shape, x_ref.dtype)[None][(0,
                              *indexer)])],
        lambda x_ref, indexer: (
            ref_addupdate(x_ref, indexer,
                                jnp.ones(x_ref.shape, x_ref.dtype)[None][(0,
                                  *indexer)])
            or [jnp.ones(x_ref.shape, x_ref.dtype)[None][(0, *indexer)]])
    ],
  )
  def test_vmap(self, ref_shape, ref_bdim, idx_shape, indexed_dims,
                    idx_bdims, out_bdim, op):

    float_ = (jnp.dtype('float64') if config.enable_x64.value else
              jnp.dtype('float32'))
    int_ = (jnp.dtype('int64') if config.enable_x64.value else
            jnp.dtype('int32'))
    axis_size = 7
    out_shape = tuple(d for d, b in zip(ref_shape, indexed_dims) if not b)
    if any(indexed_dims):
      out_shape = (*idx_shape, *out_shape)

    def maybe_insert(shape, idx):
      if idx is None:
        return shape
      return tuple_insert(shape, idx, axis_size)

    batched_ref_shape = maybe_insert(ref_shape, ref_bdim)
    ref_aval = shaped_array_ref(ref_shape, float_)
    bat_ref_aval = shaped_array_ref(batched_ref_shape, float_)

    idx_avals = [core.ShapedArray(idx_shape, int_)
                 for _ in idx_bdims]
    bat_idx_avals = [
        core.ShapedArray(maybe_insert(idx_shape, idx_bdim), int_)
        for idx_bdim in idx_bdims]

    def f(x_ref, *idxs):
      idxs_ = iter(idxs)
      indexer = tuple(next(idxs_) if b else slice(None) for b in indexed_dims)
      return op(x_ref, indexer)

    rng = self.rng()
    a = rng.randn(*bat_ref_aval.shape)
    his = [d for d, b in zip(ref_aval.shape, indexed_dims) if b]
    idxs = [rng.randint(low=0, high=hi, size=i.shape)
            for i, hi in zip(bat_idx_avals, his)]

    # discharge-of-vmap
    f_batched = jax.vmap(f, in_axes=(ref_bdim, *idx_bdims), out_axes=[out_bdim])
    stateful_jaxpr, _, stateful_consts, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f_batched), [bat_ref_aval, *bat_idx_avals])
    jaxpr, consts = discharge_state(stateful_jaxpr, stateful_consts)
    discharge_of_vmap_ans = core.eval_jaxpr(jaxpr, consts, a, *idxs)
    # vmap-of-discharge
    stateful_jaxpr, _, stateful_consts, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f), [ref_aval, *idx_avals])
    jaxpr_, consts_ = discharge_state(stateful_jaxpr, stateful_consts)
    f_batched = jax.vmap(partial(core.eval_jaxpr, jaxpr_, consts_),
                         in_axes=(ref_bdim, *idx_bdims),
                         out_axes=[out_bdim, ref_bdim])
    vmap_of_discharge_ans = f_batched(a, *idxs)

    self.assertAllClose(discharge_of_vmap_ans, vmap_of_discharge_ans,
                        check_dtypes=False)


class StateDischargeTest(jtu.JaxTestCase):

  def test_discharge_get(self):
    def f(a_ref):
      a = ref_get(a_ref, ())
      return [a + 1]
    in_avals = [shaped_array_ref((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, _ = discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 2)
    self.assertEqual(discharged_jaxpr.eqns[0].primitive, lax.add_p)
    # Should be able to evaluate this jaxpr
    self.assertListEqual(core.eval_jaxpr(discharged_jaxpr, (),
                                         jnp.float32(1.)), [2., 1.])

  def test_discharge_get_with_slice(self):
    def f(a_ref):
      a = ref_get(a_ref, (0, 1))
      return [a + 1]
    in_avals = [shaped_array_ref((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, () = discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 2)
    self.assertIn(lax.dynamic_slice_p,
                  {eqn.primitive for eqn in discharged_jaxpr.eqns})
    # Should be able to evaluate this jaxpr
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    outval, refval = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((outval == inval[0, 1] + 1).all())
    self.assertTrue((refval == inval).all())

  def test_discharge_get_with_gather(self):
    def f(a_ref):
      a = a_ref[jnp.array([0, 1])]
      return [a + 1]
    in_avals = [shaped_array_ref((4, 3), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f), in_avals)
    discharged_jaxpr, discharged_consts = discharge_state(
        stateful_jaxpr, consts)
    inval = jnp.arange(4 * 3, dtype=jnp.float32).reshape((4, 3))
    outval, refval = core.eval_jaxpr(discharged_jaxpr, discharged_consts, inval)
    self.assertTrue((outval == inval[jnp.array([0, 1])] + 1).all())
    self.assertTrue((refval == inval).all())

  def test_discharge_set(self):
    def f(a_ref, b):
      ref_set(a_ref, (), b + 1)
      return []
    in_avals = [shaped_array_ref((), jnp.dtype('float32')),
                core.ShapedArray((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that ignores the first
    # value and returns second value plus 1.
    discharged_jaxpr, _ = discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 2)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(0.),
                                     jnp.float32(1.))[0], 2.)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(2.),
                                     jnp.float32(1.))[0], 2.)

  def test_discharge_set_with_slice(self):
    def f(a_ref):
      ref_set(a_ref, (0, 1), jnp.ones(2, dtype=jnp.dtype('float32')))
      return []
    in_avals = [shaped_array_ref((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, () = discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertIn(lax.dynamic_update_slice_p,
                  {eqn.primitive for eqn in discharged_jaxpr.eqns})
    self.assertIn(lax.dynamic_slice_p,
                  {eqn.primitive for eqn in discharged_jaxpr.eqns})
    # Should be able to evaluate this jaxpr
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    refval, = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((refval == inval.at[0, 1].set(1.)).all())

  def test_discharge_set_with_gather(self):
    def f(a_ref):
      a_ref[jnp.array([0, 1])] = jnp.ones((2, 3), 'float32')
      return []
    in_avals = [shaped_array_ref((4, 3), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, discharged_consts = discharge_state(
        stateful_jaxpr, consts)
    inval = jnp.arange(4 * 3, dtype=jnp.float32).reshape((4, 3))
    refval, = core.eval_jaxpr(discharged_jaxpr, discharged_consts, inval)
    self.assertTrue((refval == inval.at[jnp.array([0, 1])].set(1.)).all())

  def test_discharge_addupdate(self):
    def f(a_ref, b):
      ref_addupdate(a_ref, (), b + 1)
      return []
    in_avals = [shaped_array_ref((), jnp.dtype('float32')),
                core.ShapedArray((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that adds the first value,
    # second value, and 1.
    discharged_jaxpr, _ = discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 2)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(0.),
                                     jnp.float32(1.))[0], 2.)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(2.),
                                     jnp.float32(1.))[0], 4.)

  def test_discharge_addupdate_with_slice(self):
    def f(a_ref):
      ref_addupdate(a_ref, (0, 1),
                             jnp.ones(2, dtype=jnp.dtype('float32')))
      return []
    in_avals = [shaped_array_ref((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, _ = discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertIn(lax.dynamic_update_slice_p,
                  {eqn.primitive for eqn in discharged_jaxpr.eqns})
    self.assertIn(lax.add_p,
                  {eqn.primitive for eqn in discharged_jaxpr.eqns})
    self.assertIn(lax.dynamic_slice_p,
                  {eqn.primitive for eqn in discharged_jaxpr.eqns})
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    refval, = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((refval == inval.at[0, 1].add(1.)).all())

  def test_discharge_addupdate_with_gather(self):
    def f(a_ref):
      ref_addupdate(a_ref, (jnp.array([0, 1]),),
                          jnp.ones((2, 3), 'float32'))
      return []
    in_avals = [shaped_array_ref((4, 3), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, discharged_consts = discharge_state(
        stateful_jaxpr, consts)
    inval = jnp.arange(4 * 3, dtype=jnp.float32).reshape((4, 3))
    refval, = core.eval_jaxpr(discharged_jaxpr, discharged_consts, inval)
    self.assertTrue((refval == inval.at[jnp.array([0, 1])].add(1.)).all())

  def test_discharge_jaxpr_with_multiple_outputs(self):
    def f(a_ref):
      a = ref_get(a_ref, ())
      b = a + 1
      return [a, b]
    in_avals = [shaped_array_ref((4,), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, _ = discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 3)
    inval = jnp.arange(4., dtype=jnp.float32)
    a, b, refval = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((a == inval).all())
    self.assertTrue((b == inval + 1).all())
    self.assertTrue((refval == inval).all())

  def test_partially_discharging_jaxpr_keeps_refs(self):
    def f(a_ref, b_ref):
      ref_set(a_ref, (), jnp.ones(4, jnp.float32))
      ref_set(b_ref, (), jnp.ones(4, jnp.float32))
      return []
    in_avals = [
        shaped_array_ref((4,), jnp.dtype('float32')),
        shaped_array_ref((4,), jnp.dtype('float32'))
        ]
    stateful_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, _ = discharge_state(
        stateful_jaxpr, consts, should_discharge=[False, True])
    self.assertLen(discharged_jaxpr.invars, 2)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertIsInstance(discharged_jaxpr.invars[0].aval, AbstractRef)
    self.assertIsInstance(discharged_jaxpr.invars[1].aval, core.ShapedArray)
    self.assertEqual(discharged_jaxpr.effects,
        {WriteEffect(len(discharged_jaxpr.constvars))})

  def test_ellipsis_index(self):
    def f(ref):
      ref_set(ref, ..., jnp.array(0., dtype=jnp.float32))
      ref_get(ref, ...)
      ref[...] = jnp.array(0., dtype=jnp.float32)
      ref[...]
      return []

    in_avals = [shaped_array_ref((), jnp.float32)]
    pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), in_avals)

  def test_partial_discharge(self):
    def f(a_ref, b_ref):
      a_ref[...] = jnp.array(0., dtype=jnp.float32)
      b_ref[...] = jnp.array(1., dtype=jnp.float32)
      return a_ref[...], b_ref[...]

    scalar_ref = shaped_array_ref((), jnp.float32)
    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f), [scalar_ref, scalar_ref])

    discharged_jaxpr, _ = discharge_state(jaxpr, (), should_discharge=[False, True])
    self.assertEqual(discharged_jaxpr.eqns[0].primitive, swap_p)
    self.assertEqual(discharged_jaxpr.eqns[1].primitive, get_p)

  def test_partial_fori_discharge(self):
    def f(a_ref, b_ref):
      def body(i, st):
        a_ref[...] += 2 * i
        b_ref[...] += i
        return ()
      lax.fori_loop(0, 5, body, init_val=())
      return a_ref[...], b_ref[...]

    ref = lambda x: AbstractRef(core.raise_to_shaped(core.get_aval(x)))
    f_jaxpr = jax.make_jaxpr(f)(ref(1.), ref(2.))
    jaxpr, _ = discharge_state(f_jaxpr.jaxpr, (), should_discharge=[False, True])
    # Effects on y_ref were discharged away but not the effects on x_ref
    self.assertEqual(f_jaxpr.effects, {ReadEffect(0), WriteEffect(0), ReadEffect(1), WriteEffect(1)})
    self.assertEqual(jaxpr.effects, {ReadEffect(0), WriteEffect(0)})
    # x_ref arg is still a reference but y_ref is discharged
    self.assertNotIsInstance(jaxpr.invars[1].aval, AbstractRef)
    self.assertIsInstance(jaxpr.invars[0].aval, AbstractRef)
    # x_ref value is returned as part of the discharged refs set.
    self.assertLen(f_jaxpr.out_avals, 2)
    self.assertLen(jaxpr.outvars, 3)


if CAN_USE_HYPOTHESIS:

  def index_arrays(size, idx_shape):
    valid_idx = hps.integers(min_value=-size, max_value=size - 1)
    return hnp.arrays(np.int32, idx_shape, elements=valid_idx)

  Shape = tuple[int, ...]

  class IndexParam(NamedTuple):
    ref_aval: shaped_array_ref
    ref_shape: Shape
    indexed_dims: list[bool]
    idx_avals: tuple[core.ShapedArray, ...]
    idx_shape: Shape
    slice_aval: core.ShapedArray
    slice_shape: Shape

  @hps.composite
  def index_params(draw):
    ref_shape = draw(hnp.array_shapes(max_dims=4, max_side=7), label='ref_shape')
    indexed_dims = draw(hps.lists(hps.booleans(),
                                  min_size=len(ref_shape),
                                  max_size=len(ref_shape)))
    idx_shape = draw(hnp.array_shapes(max_dims=3, max_side=5))
    if any(indexed_dims):
      sliced_shape = (s for s, b in zip(ref_shape, indexed_dims) if not b)
      slice_shape = (*idx_shape, *sliced_shape)
    else:
      slice_shape = ref_shape
    ref_aval = shaped_array_ref(ref_shape, np.float32)
    idx_avals = tuple(core.ShapedArray(idx_shape, np.int32) for _ in
        range(sum(indexed_dims)))
    slice_aval = core.ShapedArray(slice_shape, np.float32)
    return IndexParam(ref_aval, ref_shape, indexed_dims, idx_avals, idx_shape,
                      slice_aval, slice_shape)

  class VmappableIndexParam(NamedTuple):
    index_param: IndexParam
    ref_bdim: int | None
    non_slice_idx_bdims: tuple[int | None, ...]
    slice_bdim: int
    bat_ref_aval: shaped_array_ref
    bat_ref_shape: Shape
    bat_non_slice_idx_avals: tuple[core.ShapedArray, ...]
    bat_non_slice_idx_shapes: tuple[Shape, ...]
    bat_slice_aval: core.ShapedArray
    bat_slice_shape: Shape

  def maybe_tuple_insert(t: tuple[Any, ...], idx: int | None,
                         val: Any) -> tuple[Any, ...]:
    if idx is None:
      return t
    return tuple_insert(t, idx, val)

  @hps.composite
  def vmappable_index_params(draw, *, op_type: str):
    axis_size = draw(hps.integers(min_value=1, max_value=7), label='axis_size')
    index_param: IndexParam = draw(index_params())
    non_slice_idx_bdims = tuple(
        draw(hps.one_of(
          hps.none(),
          hps.integers(min_value=0, max_value=len(index_param.idx_shape))))
        for b in index_param.indexed_dims if b)
    bat_non_slice_idx_shapes = tuple(
        maybe_tuple_insert(index_param.idx_shape, idx_bdim, axis_size)
        for idx_bdim in non_slice_idx_bdims)
    if op_type == "swap":
      # In a swap, the ref *must* be batched
      ref_bdim = draw(hps.integers(min_value=0,
                                   max_value=len(index_param.ref_shape)))
      if any(idx_bdim is not None for idx_bdim in non_slice_idx_bdims):
        # If it's a swap, if indices are batched, val must be batched.
        slice_bdim = draw(hps.integers(
          min_value=0, max_value=len(index_param.slice_shape)))
      else:
        slice_bdim = draw(hps.one_of(hps.none(), hps.integers(
          min_value=0, max_value=len(index_param.slice_shape))))
    elif op_type == "get":
      # In a get, the indices must be batched or ref is batched
      if all(idx_bdim is None for idx_bdim in non_slice_idx_bdims):
        ref_bdim = draw(hps.integers(min_value=0,
                                     max_value=len(index_param.ref_shape)))
      else:
        ref_bdim = draw(hps.one_of(hps.none(),
          hps.integers(min_value=0, max_value=len(index_param.ref_shape))))
      slice_bdim = draw(hps.integers(
        min_value=0, max_value=len(index_param.slice_shape)))

    bat_ref_shape = maybe_tuple_insert(index_param.ref_shape, ref_bdim, axis_size)
    bat_ref_aval = shaped_array_ref(bat_ref_shape, np.float32)
    bat_non_slice_idx_avals = tuple(
        core.ShapedArray(shape, np.int32) for shape in bat_non_slice_idx_shapes)
    bat_slice_shape = maybe_tuple_insert(index_param.slice_shape, slice_bdim, axis_size)
    bat_slice_aval = core.ShapedArray(bat_slice_shape, np.float32)
    return VmappableIndexParam(index_param, ref_bdim, non_slice_idx_bdims,
                               slice_bdim, bat_ref_aval, bat_ref_shape,
                               bat_non_slice_idx_avals, bat_non_slice_idx_shapes,
                               bat_slice_aval, bat_slice_shape)

  class GetVmapParams(NamedTuple):
    vmap_index_param: VmappableIndexParam
    bat_ref: np.ndarray
    bat_idxs: tuple[np.ndarray, ...]

  @hps.composite
  def get_vmap_params(draw):
    vmap_index_param: VmappableIndexParam = draw(
        vmappable_index_params(op_type="get"))
    bat_ref = draw(hnp.arrays(np.float32, vmap_index_param.bat_ref_shape))
    bat_idx_shapes_ = iter(vmap_index_param.bat_non_slice_idx_shapes)
    bat_idxs = tuple(
        draw(index_arrays(size, next(bat_idx_shapes_)))
        for size, indexed in zip(
          vmap_index_param.index_param.ref_shape,
          vmap_index_param.index_param.indexed_dims)
        if indexed)
    assert next(bat_idx_shapes_, None) is None
    return GetVmapParams(vmap_index_param, bat_ref, bat_idxs)

  class SetVmapParams(NamedTuple):
    vmap_index_param: VmappableIndexParam
    bat_ref: np.ndarray
    bat_val: np.ndarray
    bat_idxs: tuple[np.ndarray, ...]

  @hps.composite
  def set_vmap_params(draw):
    vmap_index_param: VmappableIndexParam = draw(vmappable_index_params(
      op_type="swap"))
    bat_ref = draw(hnp.arrays(np.float32, vmap_index_param.bat_ref_shape))
    bat_idx_shapes_ = iter(vmap_index_param.bat_non_slice_idx_shapes)
    bat_idxs = tuple(
        draw(index_arrays(size, next(bat_idx_shapes_)))
        for size, indexed in zip(
          vmap_index_param.index_param.ref_shape,
          vmap_index_param.index_param.indexed_dims)
        if indexed)
    assert next(bat_idx_shapes_, None) is None
    bat_val = draw(hnp.arrays(np.float32, vmap_index_param.bat_slice_shape))
    return SetVmapParams(vmap_index_param, bat_ref, bat_val, bat_idxs)

  Indexer = tuple[Union[int, slice, np.ndarray]]

  def _unpack_idx(idx: Indexer
      ) -> tuple[Sequence[int | np.ndarray], Sequence[bool]]:
    indexed_dims = [type(i) != slice for i in idx]
    non_slice_idx = [i for i, b in zip(idx, indexed_dims) if b]
    return non_slice_idx, indexed_dims

  def _pack_idx(non_slice_idx: Sequence[int | np.ndarray],
      indexed_dims: Sequence[bool]) -> Indexer:
    idx_ = iter(non_slice_idx)
    idx = tuple(next(idx_) if b else slice(None) for b in indexed_dims)
    assert next(idx_, None) is None
    return idx

  class StateHypothesisTest(jtu.JaxTestCase):

    @hp.given(get_vmap_params())
    @hp.settings(deadline=None, print_blob=True,
                 max_examples=jtu.NUM_GENERATED_CASES.value)
    def test_get_vmap(self, get_vmap_param: GetVmapParams):

      indexed_dims = get_vmap_param.vmap_index_param.index_param.indexed_dims

      def f(ref, *non_slice_idx):
        idx = _pack_idx(non_slice_idx, indexed_dims)
        return [ref_get(ref, idx)]
      ref_aval = get_vmap_param.vmap_index_param.index_param.ref_aval
      bat_ref_aval = get_vmap_param.vmap_index_param.bat_ref_aval
      bat_non_slice_idx_avals = get_vmap_param.vmap_index_param.bat_non_slice_idx_avals
      ref_bdim = get_vmap_param.vmap_index_param.ref_bdim
      idx_bdims = get_vmap_param.vmap_index_param.non_slice_idx_bdims
      out_bdim = get_vmap_param.vmap_index_param.slice_bdim
      non_slice_idx = get_vmap_param.bat_idxs
      idx_avals = get_vmap_param.vmap_index_param.index_param.idx_avals
      ref = get_vmap_param.bat_ref

      f_batched = jax.vmap(f, in_axes=(ref_bdim, *idx_bdims), out_axes=[out_bdim])
      stateful_jaxpr, _, stateful_consts, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f_batched), [bat_ref_aval, *bat_non_slice_idx_avals])
      jaxpr, consts = discharge_state(stateful_jaxpr, stateful_consts)
      discharge_of_vmap_ans = core.eval_jaxpr(jaxpr, consts, ref, *non_slice_idx)

      # vmap-of-discharge
      stateful_jaxpr, _, stateful_consts, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval, *idx_avals])
      jaxpr_, consts_ = discharge_state(stateful_jaxpr, stateful_consts)
      f_batched = jax.vmap(partial(core.eval_jaxpr, jaxpr_, consts_),
                           in_axes=(ref_bdim, *idx_bdims),
                           out_axes=[out_bdim, ref_bdim])
      vmap_of_discharge_ans = f_batched(ref, *non_slice_idx)

      self.assertAllClose(discharge_of_vmap_ans, vmap_of_discharge_ans,
                          check_dtypes=False)


    @hp.given(set_vmap_params())
    @hp.settings(deadline=None, print_blob=True,
                 max_examples=jtu.NUM_GENERATED_CASES.value)
    def test_set_vmap(self, set_vmap_param: SetVmapParams):
      if jtu.test_device_matches(["gpu"]):
        self.skipTest("Scatter is nondeterministic on GPU")
      indexed_dims = set_vmap_param.vmap_index_param.index_param.indexed_dims

      def f(ref, val, *non_slice_idx):
        idx = _pack_idx(non_slice_idx, indexed_dims)
        ref_set(ref, idx, val)
        return []
      ref_aval = set_vmap_param.vmap_index_param.index_param.ref_aval
      bat_ref_aval = set_vmap_param.vmap_index_param.bat_ref_aval
      bat_non_slice_idx_avals = set_vmap_param.vmap_index_param.bat_non_slice_idx_avals
      ref_bdim = set_vmap_param.vmap_index_param.ref_bdim
      idx_bdims = set_vmap_param.vmap_index_param.non_slice_idx_bdims
      non_slice_idx = set_vmap_param.bat_idxs
      idx_avals = set_vmap_param.vmap_index_param.index_param.idx_avals
      ref = set_vmap_param.bat_ref
      val = set_vmap_param.bat_val
      bat_val_aval = set_vmap_param.vmap_index_param.bat_slice_aval
      val_aval = set_vmap_param.vmap_index_param.index_param.slice_aval
      val_bdim = set_vmap_param.vmap_index_param.slice_bdim

      f_batched = jax.vmap(f, in_axes=(ref_bdim, val_bdim, *idx_bdims),
                           out_axes=[])
      stateful_jaxpr, _, stateful_consts, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f_batched), [bat_ref_aval, bat_val_aval, *bat_non_slice_idx_avals])
      jaxpr, consts = discharge_state(stateful_jaxpr, stateful_consts)
      discharge_of_vmap_ans = core.eval_jaxpr(jaxpr, consts, ref, val, *non_slice_idx)

      # vmap-of-discharge
      stateful_jaxpr, _, stateful_consts, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval, val_aval, *idx_avals])
      jaxpr_, consts_ = discharge_state(stateful_jaxpr, stateful_consts)
      f_batched = jax.vmap(partial(core.eval_jaxpr, jaxpr_, consts_),
                           in_axes=(ref_bdim, val_bdim, *idx_bdims),
                           out_axes=[ref_bdim])
      vmap_of_discharge_ans = f_batched(ref, val, *non_slice_idx)

      self.assertAllClose(discharge_of_vmap_ans, vmap_of_discharge_ans,
                          check_dtypes=False)


    @hp.given(set_vmap_params())
    @hp.settings(deadline=None, print_blob=True,
                 max_examples=jtu.NUM_GENERATED_CASES.value)
    def test_addupdate_vmap(self, set_vmap_param: SetVmapParams):

      indexed_dims = set_vmap_param.vmap_index_param.index_param.indexed_dims

      def f(ref, val, *non_slice_idx):
        idx = _pack_idx(non_slice_idx, indexed_dims)
        ref_addupdate(ref, idx, val)
        return []
      ref_aval = set_vmap_param.vmap_index_param.index_param.ref_aval
      bat_ref_aval = set_vmap_param.vmap_index_param.bat_ref_aval
      bat_non_slice_idx_avals = set_vmap_param.vmap_index_param.bat_non_slice_idx_avals
      ref_bdim = set_vmap_param.vmap_index_param.ref_bdim
      idx_bdims = set_vmap_param.vmap_index_param.non_slice_idx_bdims
      non_slice_idx = set_vmap_param.bat_idxs
      idx_avals = set_vmap_param.vmap_index_param.index_param.idx_avals
      ref = set_vmap_param.bat_ref
      val = set_vmap_param.bat_val
      bat_val_aval = set_vmap_param.vmap_index_param.bat_slice_aval
      val_aval = set_vmap_param.vmap_index_param.index_param.slice_aval
      val_bdim = set_vmap_param.vmap_index_param.slice_bdim

      f_batched = jax.vmap(f, in_axes=(ref_bdim, val_bdim, *idx_bdims),
                           out_axes=[])
      stateful_jaxpr, _, stateful_consts, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f_batched), [bat_ref_aval, bat_val_aval, *bat_non_slice_idx_avals])
      jaxpr, consts = discharge_state(stateful_jaxpr, stateful_consts)
      discharge_of_vmap_ans = core.eval_jaxpr(jaxpr, consts, ref, val, *non_slice_idx)

      # vmap-of-discharge
      stateful_jaxpr, _, stateful_consts, () = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval, val_aval, *idx_avals])
      jaxpr_, consts_ = discharge_state(stateful_jaxpr, stateful_consts)
      f_batched = jax.vmap(partial(core.eval_jaxpr, jaxpr_, consts_),
                           in_axes=(ref_bdim, val_bdim, *idx_bdims),
                           out_axes=[ref_bdim])
      vmap_of_discharge_ans = f_batched(ref, val, *non_slice_idx)

      self.assertAllClose(discharge_of_vmap_ans, vmap_of_discharge_ans,
                          check_dtypes=False)


class StateControlFlowTest(jtu.JaxTestCase):

  def test_simple_cond(self):
    def f(pred):
      def body(x_ref):
        def true_fun():
          x_ref[()] = 1.
        def false_fun():
          pass
        lax.cond(pred, true_fun, false_fun)
      return for_loop.run_state(body, 0.)
    jaxpr = jax.make_jaxpr(f)(True).jaxpr
    self.assertEmpty(jaxpr.effects)
    self.assertAllClose(jax.jit(f)(True), 1.)
    self.assertAllClose(jax.jit(f)(False), 0.)

  def test_simple_cond_with_return(self):
    def f(pred):
      def body(refs):
        x_ref, y_ref = refs
        def true_fun():
          x_ref[()] = 1.
          return 4.
        def false_fun():
          return 5.
        out = lax.cond(pred, true_fun, false_fun)
        y_ref[...] = out
      return for_loop.run_state(body, (0., 0.))
    jaxpr = jax.make_jaxpr(f)(True).jaxpr
    self.assertEmpty(jaxpr.effects)
    out = jax.jit(f)(True)
    self.assertTupleEqual(out, (1., 4.))
    out = jax.jit(f)(False)
    self.assertTupleEqual(out, (0., 5.))

  def test_cond_discharge(self):
    def f0(pred, x_ref, y_ref):
      def true_fun():
        x_ref[...] = 1.
      def false_fun():
        y_ref[...] = 2.
      lax.cond(pred, true_fun, false_fun)
      return x_ref[...], y_ref[...]
    ref = lambda x: AbstractRef(core.raise_to_shaped(core.get_aval(x)))
    f_jaxpr = jax.make_jaxpr(f0)(False, ref(3.), ref(4.))
    jaxpr, _ = discharge_state(f_jaxpr.jaxpr, (), should_discharge=[False, False, True])
    # Effects on y_ref were discharged away but not the effects on x_ref
    self.assertEqual(f_jaxpr.effects, {ReadEffect(1), WriteEffect(1), ReadEffect(2), WriteEffect(2)})
    self.assertEqual(jaxpr.effects, {ReadEffect(1), WriteEffect(1)})
    # x_ref arg is still a reference but y_ref is discharged
    self.assertNotIsInstance(jaxpr.invars[2].aval, AbstractRef)
    self.assertIsInstance(jaxpr.invars[1].aval, AbstractRef)
    # x_ref value is returned as part of the discharged refs set.
    self.assertLen(f_jaxpr.out_avals, 2)
    self.assertLen(jaxpr.outvars, 3)

  def test_cond_with_ref_reuse(self):
    def f(pred):
      def body(x_ref):
        def true_fun():
          x_ref[()] = 1.
        def false_fun():
          x_ref[()] = 2.
        lax.cond(pred, true_fun, false_fun)
      return run_state(body)(0.)
    jaxpr = jax.make_jaxpr(f)(True).jaxpr
    self.assertEmpty(jaxpr.effects)
    out_true = jax.jit(f)(True)
    expected_true = 1.
    self.assertAllClose(out_true, expected_true)
    out_false = jax.jit(f)(False)
    expected_false = 2.
    self.assertAllClose(out_false, expected_false)

  def test_simple_cond_using_multiple_refs_with_interleaved_consts(self):
    def f(pred):
      def body(refs):
        x_ref, y_ref, z_ref = refs
        a = jnp.array(2.)
        b = jnp.array(2.)
        def true_fun():
          x_ref[()] = 1. * a
          y_ref[()] = 1.
        def false_fun():
          z_ref[()] = 1.
          x_ref[()] = 2.
          x_ref[()] = 2. * b
        lax.cond(pred, true_fun, false_fun)
      return run_state(body)((0., 0., 0.))
    jaxpr = jax.make_jaxpr(f)(True).jaxpr
    self.assertEmpty(jaxpr.effects)
    out_true = jax.jit(f)(True)
    expected_true = (2.,  1., 0.)
    self.assertAllClose(out_true, expected_true)
    out_false = jax.jit(f)(False)
    expected_false = (4., 0., 1.)
    self.assertAllClose(out_false, expected_false)

  def test_cond_nested_run_state_using_multiple_refs_with_interleaved_consts(
      self):
    def f(pred):
      @run_state
      def body(refs):
        x_ref, o_ref1, o_ref2 = refs
        @run_state
        def body2(refs):
          y_ref, o_ref3 = refs
          @run_state
          def body3(z_ref):
            a = jnp.array(2.)
            b = jnp.array(2.)
            def true_fun():
              x_ref[()] = 1. * a
              y_ref[()] = 1.
            def false_fun():
              z_ref[()] = 1.
              x_ref[()] = 2.
              x_ref[()] = 2. * b
            lax.cond(pred, true_fun, false_fun)
          o_ref3[...] = body3(0.)
        o_ref1[...], o_ref2[...] = body2((0., 0.))
      return body((0., 0., 0.))
    jaxpr = jax.make_jaxpr(f)(True).jaxpr
    self.assertEmpty(jaxpr.effects)
    out_true = jax.jit(f)(True)
    expected_true = (2.,  1., 0.)
    self.assertAllClose(out_true, expected_true)
    out_false = jax.jit(f)(False)
    expected_false = (4., 0., 1.)
    self.assertAllClose(out_false, expected_false)

  def test_nested_cond_using_multiple_refs_with_interleaved_consts(self):
    def f(pred1, pred2):
      def body(refs):
        x_ref, y_ref, z_ref = refs
        a = jnp.array(2.)
        def true_fun():
          b = jnp.array(2.)
          def inner_true_fun():
            x_ref[()] = 1. * a
            y_ref[()] = 1.
          def inner_false_fun():
            z_ref[()] = 1.
            x_ref[()] = 2.
            x_ref[()] = 2. * b
          lax.cond(pred2, inner_true_fun, inner_false_fun)
        def false_fun():
          a = jnp.array(2.)
          b = jnp.array(2.)
          def inner_true_fun():
            x_ref[()] = 1. * a
            z_ref[()] = 4.
          def inner_false_fun():
            x_ref[()] = 2. * b
          lax.cond(pred2, inner_true_fun, inner_false_fun)
        lax.cond(pred1, true_fun, false_fun)
      return run_state(body)((0., 0., 0.))
    jaxpr = jax.make_jaxpr(f)(True, False).jaxpr
    self.assertEmpty(jaxpr.effects)
    out = jax.jit(f)(True, True)
    expected = (2.,  1., 0.)
    self.assertTupleEqual(out, expected)
    out = jax.jit(f)(True, False)
    expected = (4., 0., 1.)
    self.assertTupleEqual(out, expected)
    out = jax.jit(f)(False, True)
    expected = (2., 0., 4.)
    self.assertTupleEqual(out, expected)
    out = jax.jit(f)(False, False)
    expected = (4., 0., 0.)
    self.assertTupleEqual(out, expected)

  def test_nested_cond(self):
    def f(pred):
      def body(x_ref):
        def true_fun():
          def true_fun_inner():
            x_ref[()] = 1.
          def false_fun_inner():
            pass
          return lax.cond(pred, true_fun_inner, false_fun_inner)
        def false_fun():
          pass
        lax.cond(pred, true_fun, false_fun)
      return for_loop.run_state(body, 0.)
    jaxpr = jax.make_jaxpr(f)(True).jaxpr
    self.assertEmpty(jaxpr.effects)
    self.assertAllClose(jax.jit(f)(True), 1.)
    self.assertAllClose(jax.jit(f)(False), 0.)

  def test_cond_jvp_with_state(self):
    def f(pred, init_value):
      def body(x_ref):
        def true_fun():
          x_ref[()] = x_ref[()] ** 2
        def false_fun():
          pass
        lax.cond(pred, true_fun, false_fun)
      return for_loop.run_state(body, init_value)

    out_primal, out_tangent = jax.jvp(partial(f, True), (3.,), (1.,))
    self.assertAllClose(out_primal, 9.)
    self.assertAllClose(out_tangent, 6.)

    out_primal, out_tangent = jax.jvp(partial(f, False), (3.,), (1.,))
    self.assertAllClose(out_primal, 3.)
    self.assertAllClose(out_tangent, 1.)

  def test_cond_vmap_not_implemented(self):
    @jax.jit
    def f(init_value):
      def body(x_ref):
        def true_fun():
          x_ref[()] = x_ref[()] ** 2
        def false_fun():
          pass
        lax.cond(x_ref[()] < 1, true_fun, false_fun)
      return for_loop.run_state(body, init_value)

    with self.assertRaises(NotImplementedError):
      jax.vmap(f)(jnp.arange(2.))

  def test_cond_grad_not_implemented(self):
    @jax.jit
    def f(init_value):
      def body(x_ref):
        def true_fun():
          x_ref[()] = x_ref[()] ** 2
        def false_fun():
          pass
        lax.cond(True, true_fun, false_fun)
      return for_loop.run_state(body, init_value)

    with self.assertRaises(NotImplementedError):
      jax.grad(f)(3.)

  def test_while_with_state_in_body(self):
    def f(x, y, z):
      @run_state
      def body(x_ref):
        def cond(i):
          return i < y
        def body(i):
          x_ref[...] += z
          return i + 1
        lax.while_loop(cond, body, 0)
      return body(x)
    jaxpr = jax.make_jaxpr(f)(0, 5, 2).jaxpr
    self.assertEmpty(jaxpr.effects)
    self.assertAllClose(jax.jit(f)(0, 5, 2), 10)
    self.assertAllClose(jax.jit(f)(1, 2, 3), 7)

  def test_scan_with_state_in_body(self):
    def f(x, w, y, zs):
      @run_state
      def body(refs):
        x_ref, w_ref = refs
        def body(y, z):
          x_ref[...] += y
          w_ref[...] += z
          return y + 1, ()
        lax.scan(body, y, zs)
      return body((x, w))
    zs = jnp.arange(5)
    jaxpr = jax.make_jaxpr(f)(0, 1, 5, zs).jaxpr
    self.assertEmpty(jaxpr.effects)
    self.assertAllClose(jax.jit(f)(0, 1, 5, zs), (35, 11))
    self.assertAllClose(jax.jit(f)(1, 1, 2, zs), (21, 11))

  def test_scan_with_state_in_body_nested(self):
    @run_state
    def g(refs):
      a_ref, x_ref, w_ref, y_ref, zs_ref = refs

      def f(x, w, y, zs):
        @run_state
        def loop(refs):
          x_ref, w_ref = refs
          def body(y, z):
            x_ref[...] += y
            w_ref[...] += z
            a_ref[...] += z
            return y + 1, ()
          lax.scan(body, y, zs)
        a_ref[...] += 1
        out = loop((x, w))
        a_ref[...] += 1
        return out

      x, w = f(x_ref[...], w_ref[...], y_ref[...], zs_ref[...])
      x_ref[...] = x
      w_ref[...] = w

    zs = jnp.arange(5)
    jaxpr = jax.make_jaxpr(g)((1, 0, 1, 5, zs)).jaxpr
    self.assertEmpty(jaxpr.effects)
    self.assertAllClose(jax.jit(g)((1, 0, 1, 5, zs))[:3], (13, 35, 11))
    self.assertAllClose(jax.jit(g)((1, 1, 1, 2, zs))[:3], (13, 21, 11))


class GeneralRefTest(jtu.JaxTestCase):

  def test_unshaped_ref(self):
    def f(x_ref):
      x = x_ref[...]
      x_ref[...] = x
      ref_addupdate(x_ref, (), x)
      return [x]
    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f), [AbstractRef(core.UnshapedArray(jnp.int32))])
    self.assertIs(type(jaxpr.outvars[0].aval), core.UnshapedArray)
    self.assertEqual(jaxpr.outvars[0].aval.dtype, jnp.dtype("int32"))

  def test_token(self):
    def f(x_ref):
      x = x_ref[...]
      x_ref[...] = x
      ref_addupdate(x_ref, (), x)
      return [x]
    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f), [AbstractRef(core.AbstractToken())])
    self.assertIs(type(jaxpr.outvars[0].aval), core.AbstractToken)

  def test_ref_of_ref(self):
    def f(x_ref_ref):
      x_ref = x_ref_ref[...]
      return [x_ref]
    # Not sure why you'd ever want to do this, but it works!
    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f),
        [AbstractRef(AbstractRef(core.ShapedArray((), jnp.int32)))])
    self.assertIs(type(jaxpr.outvars[0].aval), AbstractRef)
    self.assertIs(type(jaxpr.outvars[0].aval.inner_aval), core.ShapedArray)


class RunStateTest(jtu.JaxTestCase):

  def test_simple_run_state(self):
    out = run_state(lambda _: None)(1)
    self.assertEqual(out, 1)

  def test_nontrivial_run_state(self):
    def f(refs):
      x_ref, y_ref = refs
      x = x_ref[...] * y_ref[...]
      y_ref[...] = x * 2
      x_ref[...] = y_ref[...] + x_ref[...]
      # x + x * y * 2, x * y * 2
    x, y = run_state(f)((2, 3))
    self.assertEqual(x, 2 + 2 * 3 * 2)
    self.assertEqual(y, 2 * 3 * 2)

  def test_nontrivial_run_state_jit(self):
    def f(refs):
      x_ref, y_ref = refs

      @jax.jit
      def g():
        x = x_ref[...] * y_ref[...]
        y_ref[...] = x * 2
        x_ref[...] = y_ref[...] + x_ref[...]
        # x + x * y * 2, x * y * 2

      g()

    x, y = run_state(f)((2, 3))
    self.assertEqual(x, 2 + 2 * 3 * 2)
    self.assertEqual(y, 2 * 3 * 2)

  def test_simple_run_state_with_multiple_refs(self):
    out1, out2 = run_state(lambda _: None)((1, 2))
    self.assertEqual(out1, 1)
    self.assertEqual(out2, 2)

  def test_simple_run_state_with_tuple(self):
    out1, out2 = run_state(lambda _: None)((1, 2))
    self.assertEqual(out1, 1)
    self.assertEqual(out2, 2)

  def test_can_stage_run_state(self):
    def f(x):
      return run_state(lambda _: None)(x)
    _ = jax.make_jaxpr(f)(2)

  def test_nested_run_state_captures_effects(self):
    def f(x):
      def body(x_ref):
        def inner(y_ref):
          y_ref[...]
          x_ref[...]
        run_state(inner)(1)
      return run_state(body)(x)
    jaxpr = jax.make_jaxpr(f)(2)
    self.assertEmpty(jaxpr.effects)
    self.assertEmpty(jaxpr.jaxpr.eqns[0].effects)
    self.assertSetEqual(jaxpr.jaxpr.eqns[0].params["jaxpr"].effects,
                        {ReadEffect(0)})
    self.assertSetEqual(
        jaxpr.jaxpr.eqns[0].params["jaxpr"].eqns[0].params["jaxpr"].effects,
                        {ReadEffect(0), ReadEffect(1)})

  def test_jvp_of_run_state(self):
    @run_state
    def f(refs):
      x_ref, y_ref = refs
      y_ref[...] = jnp.sin(x_ref[...])
    xy, xy_t = jax.jvp(f, ((2., 1.),), ((3., 1.),))
    # x, sin(x)
    self.assertAllClose(xy, (2., np.sin(2.)))
    # t, cos(x) * t
    self.assertAllClose(xy_t, (3., 3 * np.cos(2.)))

    x, x_t = jax.jvp(lambda x: f((x, 0.))[1], (2.,), (3.,))
    self.assertAllClose(x, np.sin(2.))
    self.assertAllClose(x_t, 3 * np.cos(2.))

  def test_jvp_of_run_state_with_zero_tangent(self):
    @run_state
    def f(refs):
      x_ref, z_ref, y_ref = refs
      del z_ref
      y_ref[...] = jnp.sin(x_ref[...])
    x, x_t = jax.jvp(lambda x: f((x, 0., 0.,))[2], (2.,), (3.,))
    self.assertAllClose(x, np.sin(2.))
    self.assertAllClose(x_t, 3 * np.cos(2.))

  def test_linearize_of_run_state(self):
    @run_state
    def f(refs):
      x_ref, y_ref = refs
      y_ref[...] = jnp.sin(x_ref[...])

    (x, y), f_lin = jax.linearize(f, (1., 0.))
    self.assertAllClose(x, 1.)
    self.assertAllClose(y, np.sin(1.))
    x_t, y_t = f_lin((2., 1.))
    self.assertAllClose(x_t, 2.)
    self.assertAllClose(y_t, 2. * np.cos(1.))

  def test_grad_of_run_state(self):
    @run_state
    def f(refs):
      x_ref, y_ref = refs
      y_ref[...] = jnp.sin(x_ref[...])

    def sin(x):
      return f((x, 0.))[1]

    x_g = jax.grad(sin)(1.)
    self.assertAllClose(x_g, np.cos(1.))

    x_g2 = jax.grad(jax.grad(sin))(1.)
    self.assertAllClose(x_g2, -np.sin(1.))

    x_g3 = jax.grad(jax.grad(jax.grad(sin)))(1.)
    self.assertAllClose(x_g3, -np.cos(1.))

  def test_vjp_of_run_state(self):
    @run_state
    def f(refs):
      x_ref, y_ref = refs
      y_ref[...] = jnp.sin(x_ref[...])

    (x, y), f_vjp = jax.vjp(f, (1., 0.))
    self.assertAllClose(x, 1.)
    self.assertAllClose(y, np.sin(1.))
    ((x_ct, y_ct),) = f_vjp((0., 1.))
    self.assertAllClose(x_ct, np.cos(1.))
    self.assertAllClose(y_ct, 0.)

  def test_vjp_of_run_state_single(self):
    @run_state
    def f(x_ref):
      x = x_ref[...]
      def _body(ref):
        ref[...] = jnp.sin(ref[...])
      x = run_state(_body)(x)
      x_ref[...] = x

    y, f_lin = jax.linearize(f, 1.)
    self.assertAllClose(y, np.sin(1.))
    y_t = f_lin(1.)
    self.assertAllClose(y_t, np.cos(1.))

    y, f_vjp = jax.vjp(f, 1.)
    self.assertAllClose(y, np.sin(1.))
    x_ct, = f_vjp(1.)
    self.assertAllClose(x_ct, np.cos(1.))

    jtu.check_grads(f, (0.5,), order=3)


if CAN_USE_HYPOTHESIS:

  class FuncSpec(NamedTuple):
    fun: Callable[..., Any]
    name: str
    min_rank: int = 0
    max_rank: int = 4
    min_dim: int = 0
    max_dim: int = 4

    def call(self, *args):
      return run_state(self.fun)(*args)

    def ref(self, *args):
      return run_state_reference(self.fun)(*args)

  def sin_stateful(refs):
    x_ref, y_ref = refs
    y_ref[...] = jnp.sin(x_ref[...])

  sin_spec = FuncSpec(sin_stateful, "sin")

  def cos_stateful(refs):
    x_ref, y_ref = refs
    y_ref[...] = jnp.cos(x_ref[...])

  cos_spec = FuncSpec(cos_stateful, "cos")

  def mul2_stateful(refs):
    x_ref, y_ref = refs
    y_ref[...] = x_ref[...]
    y_ref[...] = y_ref[...] + x_ref[...]

  mul2_spec = FuncSpec(mul2_stateful, "mul2")

  def mul2_stateful_with_constant(refs):
    x_ref, y_ref = refs
    y_ref[...] = (2. * np.ones(x_ref.shape, x_ref.dtype)) * x_ref[...]

  mul2_constant_spec = FuncSpec(mul2_stateful_with_constant, "mul2_c")

  def crazy_identity_stateful(refs):
    x_ref, y_ref = refs
    x = x_ref[...]
    x_ref[...] = (x + x) / 2
    y_ref[...] = x_ref[...]
    y = y_ref[...]
    y_ref[...] = (y + y) / 2

  crazy_identity_spec = FuncSpec(crazy_identity_stateful, "id")

  def func_spec(depth: int = 4):
    raw_specs = hps.sampled_from([sin_spec, cos_spec, mul2_spec,
                                  mul2_constant_spec, crazy_identity_spec])
    if depth > 0:
      return hps.one_of([raw_specs, nest_spec(depth - 1), add_spec(depth - 1),
                         compose_spec(depth - 1)])
    return raw_specs

  @hps.composite
  def compose_spec(draw, depth):
    f1 = draw(func_spec(depth))
    f2 = draw(func_spec(depth))
    def wrapped_impl(*args):
      f1.fun(*args)
      f2.fun(*args)
    return FuncSpec(wrapped_impl,
                    f"({f2.name} . {f1.name})",
                    min_rank=max(f1.min_rank, f2.min_rank),
                    max_rank=min(f1.max_rank, f2.max_rank),
                    min_dim=max(f1.min_dim, f2.min_dim),
                    max_dim=min(f1.max_dim, f2.max_dim))

  @hps.composite
  def nest_spec(draw, depth):
    f = draw(func_spec(depth))
    def wrapped_impl(refs):
      x_ref, y_ref = refs
      x, y = x_ref[...], y_ref[...]
      x, y = run_state(f.fun)((x, y))
      x_ref[...], y_ref[...] = x, y
    return FuncSpec(wrapped_impl,
                    f"nest({f.name})",
                    min_rank=f.min_rank,
                    max_rank=f.max_rank,
                    min_dim=f.min_dim,
                    max_dim=f.max_dim)


  @hps.composite
  def add_spec(draw, depth):
    f1 = draw(func_spec(depth))
    f2 = draw(func_spec(depth))
    def wrapped_impl(refs):
      x_ref, y_ref = refs
      x, y = x_ref[...], y_ref[...]
      x1, y1 = run_state(f1.fun)((x, y))
      x2, y2 = run_state(f2.fun)((x, y))
      x_ref[...], y_ref[...] = x1 + x2, y1 + y2
    return FuncSpec(wrapped_impl,
                    f"({f2.name} + {f1.name})",
                    min_rank=max(f1.min_rank, f2.min_rank),
                    max_rank=min(f1.max_rank, f2.max_rank),
                    min_dim=max(f1.min_dim, f2.min_dim),
                    max_dim=min(f1.max_dim, f2.max_dim))

  class RunStateHypothesisTest(jtu.JaxTestCase):

    @jax.legacy_prng_key('allow')
    @hp.given(hps.data())
    @hp.settings(deadline=None, print_blob=True,
                 max_examples=jtu.NUM_GENERATED_CASES.value)
    def test_jvp(self, data):

      spec = data.draw(func_spec())

      def impl(x):
        return spec.call((x, jnp.zeros_like(x)))[1]

      def ref(x):
        return spec.ref((x, jnp.zeros_like(x)))[1]

      k1, k2 = random.split(random.PRNGKey(0))
      shape = data.draw(hnp.array_shapes(min_dims=spec.min_rank,
                        max_dims=spec.max_rank, min_side=spec.min_dim,
                        max_side=spec.max_dim))
      x = random.normal(k1, shape)
      t = random.normal(k2, x.shape)
      y, y_t = jax.jvp(impl, (x,), (t,))
      y_ref, y_ref_t = jax.jvp(ref, (x,), (t,))
      self.assertAllClose(y, y_ref)
      self.assertAllClose(y_t, y_ref_t)

    @jax.legacy_prng_key('allow')
    @hp.given(hps.data())
    @hp.settings(deadline=None, print_blob=True,
                 max_examples=jtu.NUM_GENERATED_CASES.value)
    def test_linearize(self, data):

      spec = data.draw(func_spec())

      def impl(x):
        return spec.call((x, jnp.zeros_like(x)))[1]

      def ref(x):
        return spec.ref((x, jnp.zeros_like(x)))[1]


      k1, k2 = random.split(random.PRNGKey(0))
      shape = data.draw(hnp.array_shapes(min_dims=spec.min_rank,
                        max_dims=spec.max_rank, min_side=spec.min_dim,
                        max_side=spec.max_dim))
      x = random.normal(k1, shape)
      y, impl_lin = jax.linearize(impl, x)
      y_ref, ref_lin = jax.linearize(ref, x)
      self.assertAllClose(y, y_ref, atol=1e-2, rtol=1e-2)
      t = random.normal(k2, x.shape)
      self.assertAllClose(impl_lin(t), ref_lin(t), atol=1e-2, rtol=1e-2)

    @jax.legacy_prng_key('allow')
    @hp.given(hps.data())
    @hp.settings(deadline=None, print_blob=True,
                 max_examples=jtu.NUM_GENERATED_CASES.value)
    def test_vjp(self, data):

      spec = data.draw(func_spec())

      def impl(x):
        return spec.call((x, jnp.zeros_like(x)))[1]

      def ref(x):
        return spec.ref((x, jnp.zeros_like(x)))[1]


      key, k1, k2 = random.split(random.PRNGKey(0), 3)
      shape = data.draw(hnp.array_shapes(min_dims=spec.min_rank,
                        max_dims=spec.max_rank, min_side=spec.min_dim,
                        max_side=spec.max_dim))
      x = random.normal(k1, shape)

      # First order
      y, impl_lin = jax.linearize(impl, x)
      y_ref, ref_lin = jax.linearize(ref, x)
      self.assertAllClose(y, y_ref)
      t = random.normal(k2, x.shape)
      self.assertAllClose(impl_lin(t), ref_lin(t))

      y, impl_vjp = jax.vjp(impl, x)
      y_ref, ref_vjp = jax.vjp(ref, x)
      self.assertAllClose(y, y_ref)
      t = random.normal(jax.random.clone(k2), x.shape)
      y2 = random.normal(jax.random.clone(k1), y.shape)
      self.assertAllClose(impl_vjp(t), ref_vjp(t))

      # Second order
      key, k1, k2 = random.split(key, 3)
      t2 = random.normal(k2, t.shape)

      (x,), impl_lin2 = jax.linearize(impl_vjp, t2)
      (x_ref,), ref_lin2 = jax.linearize(ref_vjp, t2)
      self.assertAllClose(x, x_ref)
      y2 = random.normal(k1, y.shape)
      self.assertAllClose(impl_lin2(y2), ref_lin2(y2))

      (x,), impl_vjp2 = jax.vjp(impl_vjp, t2)
      (x_ref,), ref_vjp2 = jax.vjp(ref_vjp, t2)
      self.assertAllClose(x, x_ref)
      y2 = random.normal(jax.random.clone(k1), y.shape)
      self.assertAllClose(impl_vjp2((y2,)), ref_vjp2((y2,)))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
