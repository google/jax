import contextlib
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

import functools
import itertools
import math
import os
import re
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax import random
from jax._src import checkify
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.lax.control_flow.for_loop import for_loop
from jax._src.lib import version as jaxlib_version
from jax._src.pallas.pallas_call import _trace_kernel_to_jaxpr
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas import tpu as pltpu
else:
  pltpu = None


# TODO(sharadmv): Update signatures of pallas_call to correct inputs/outputs.
# pylint: disable=no-value-for-parameter

config.parse_flags_with_absl()


def smem_on_tpu():
  if jtu.test_device_matches(["tpu"]):
    return pltpu.SMEM
  else:
    return None


intx = dtypes.canonicalize_dtype(jnp.int64)
floatx = dtypes.canonicalize_dtype(jnp.float64)


@functools.partial(jax.jit, static_argnames=["bm", "bn", "gm", "bk",
                                             "interpret", "debug"])
def matmul(x, y, *, bm, bn, gm, bk, interpret, debug=False):
  m, n, k = x.shape[0], y.shape[1], x.shape[1]
  @functools.partial(
      pl.pallas_call, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      grid=pl.cdiv(m, bm) * pl.cdiv(n, bn))
  def matmul_kernel(x_ref, y_ref, o_ref):
    pid = pl.program_id(axis=0).astype(intx)
    num_pid_m = m // bm
    num_pid_n = n // bn
    num_pid_in_group = gm * num_pid_n
    group_id = lax.div(pid, num_pid_in_group)
    first_pid_m = group_id * gm
    group_size_m = jnp.minimum(num_pid_m - first_pid_m, gm)
    pid_m = first_pid_m + lax.rem(pid, group_size_m)
    pid_n = lax.div(lax.rem(pid, num_pid_in_group), group_size_m)
    idx_m = pid_m * bm + jnp.arange(bm)
    idx_n = pid_n * bn + jnp.arange(bn)
    idx_m = pl.max_contiguous(pl.multiple_of(idx_m, bm), bm)
    idx_n = pl.max_contiguous(pl.multiple_of(idx_n, bn), bn)
    acc = jnp.zeros((bm, bn), dtype=jnp.float32)
    def body(i, acc_ref):
      idx_k = i * bk + jnp.arange(bk)
      x_idx = (
          jax.lax.broadcast_in_dim(idx_m, (bm, bk), (0,)),
          jax.lax.broadcast_in_dim(idx_k, (bm, bk), (1,)))
      y_idx = (
          jax.lax.broadcast_in_dim(idx_k, (bk, bn), (0,)),
          jax.lax.broadcast_in_dim(idx_n, (bk, bn), (1,)))
      x_block, y_block = x_ref[x_idx], y_ref[y_idx]
      out = pl.dot(x_block, y_block)
      acc_ref[:, :] += out
    acc = for_loop(k // bk, body, acc).astype(o_ref.dtype)
    o_idx = (
        jax.lax.broadcast_in_dim(idx_m, (bm, bn), (0,)),
        jax.lax.broadcast_in_dim(idx_n, (bm, bn), (1,)),
        )
    o_ref[o_idx] = acc
  return matmul_kernel(x, y)


@functools.partial(jax.jit, static_argnames=["bm", "bn", "bk",
                                             "interpret", "debug"])
def matmul_block_spec(x, y, *, bm, bn, bk, interpret, debug=False):
  m, n, k = x.shape[0], y.shape[1], x.shape[1]
  @functools.partial(
      pl.pallas_call,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      in_specs=[
          pl.BlockSpec((bm, x.shape[1]), lambda i, _: (i, 0)),
          pl.BlockSpec((y.shape[0], bn), lambda _, j: (0, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
      grid=(pl.cdiv(m, bm), pl.cdiv(n, bn)),
  )
  def matmul_kernel(x_ref, y_ref, o_ref):
    acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)
    def body(i, acc_ref):
      x_block = pl.load(x_ref, (slice(None), pl.ds(i * bk, bk)))
      y_block = pl.load(y_ref, (pl.ds(i * bk, bk), slice(None)))
      acc_ref[:, :] += pl.dot(x_block, y_block)
    acc = for_loop(k // bk, body, acc).astype(o_ref.dtype)
    o_ref[:, :] = acc
  return matmul_kernel(x, y)


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jtu.test_device_matches(["cpu"]) and not self.INTERPRET:
      self.skipTest("On CPU the test works only in interpret mode")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPU with capability >= sm80")
    if sys.platform == "win32" and not self.INTERPRET:
      self.skipTest("Only works on non-Windows platforms")

    super().setUp()
    _trace_kernel_to_jaxpr.cache_clear()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


class PallasCallTest(PallasBaseTest):
  def test_add_one(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), floatx))
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1.

    x = 0.
    self.assertEqual(add_one(x), 1.)

  def test_add_singleton_vector(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
        grid=1)
    def add_one(x_ref, o_ref):
      o_ref[0] = x_ref[0] + 1.

    x = jnp.array([0.], jnp.float32)
    np.testing.assert_allclose(add_one(x), jnp.array([1.], jnp.float32))

  def test_add_vector_block_spec(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8,), intx),
        in_specs=[pl.BlockSpec((1,), lambda i: i)],
        out_specs=pl.BlockSpec((1,), lambda i: i),
        grid=8,
    )
    def add_one(x_ref, o_ref):
      o_ref[0] = x_ref[0] + 1

    np.testing.assert_allclose(add_one(jnp.arange(8)), jnp.arange(8) + 1)

  def test_add_matrix_block_spec(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 8), intx),
        in_specs=[pl.BlockSpec((2, 2), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((2, 2), lambda i, j: (i, j)),
        grid=(4, 4),
    )
    def add_one(x_ref, o_ref):
      o_ref[:, :] = x_ref[:, :] + 1

    x = jnp.arange(64).reshape((8, 8))
    np.testing.assert_allclose(add_one(x), x + 1)

  def test_bool_array(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.bool_))
    def logical_and(x_ref, o_ref):
      o_ref[()] = jnp.logical_and(x_ref[()], True)

    x = jnp.array(True)
    self.assertTrue(jnp.all(logical_and(x)))

  def test_vector_indexing(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), floatx),
        grid=1)
    def index(x_ref, i_ref, o_ref):
      o_ref[()] = x_ref[i_ref[()]]

    x = jnp.arange(5.)
    for i in range(5):
      np.testing.assert_allclose(index(x, i), x[i])

  def test_pallas_call_no_outputs(self):
    a = np.arange(256, dtype=np.int32)
    f = self.pallas_call(lambda x_ref: None, ())
    self.assertAllClose((), f(a))

  def test_pallas_call_out_shape_is_singleton_tuple(self):
    a = np.arange(1024, dtype=np.int32).reshape((8, 128))
    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=(a,))
    res = f(a)
    self.assertIsInstance(res, tuple)
    self.assertLen(res, 1)

  def test_pallas_call_out_shape_is_list(self):
    a = np.arange(1024, dtype=np.int32).reshape((8, 128))
    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=[a])
    res = f(a)
    # TODO(necula): we normalize out_shape to a tuple, we shouldn't.
    self.assertIsInstance(res, tuple)

  @jtu.skip_on_devices("gpu")  # TODO: RET_CHECK failure
  def test_block_spec_with_padding(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    def f(*, shape, block_shape):
      def kernel(o1_ref):
        assert o1_ref.shape == block_shape
        o1_ref[...] = jnp.full(o1_ref.shape, pl.program_id(0))

      return self.pallas_call(kernel,
                              jax.ShapeDtypeStruct(shape, dtype=np.int32),
                              grid=((shape[0] + block_shape[0] - 1) // block_shape[0],),
                              out_specs=pl.BlockSpec(block_shape, lambda i: i))()
    # No padding
    pids = f(shape=(8,), block_shape=(2,))
    self.assertAllClose(pids,
                        np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32))
    # Pad the last block
    pids = f(shape=(8,), block_shape=(3,))
    self.assertAllClose(pids,
                        np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32))
    # Works even if the shape is smaller than 1 block
    pids = f(shape=(3,), block_shape=(8,))
    self.assertAllClose(pids,
                        np.array([0, 0, 0], dtype=np.int32))

  @parameterized.parameters("int32", "float32")
  def test_block_spec_padding_is_nan(self, dtype_name):
    if not self.INTERPRET:
      self.skipTest("Only applicable for the interpret mode")

    dtype = np.dtype(dtype_name)
    def copy_kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    res = self.pallas_call(copy_kernel,
                           jax.ShapeDtypeStruct((6,), dtype=dtype),
                           grid=(1,),
                           in_specs=[pl.BlockSpec((6,), lambda i: 0)])(
        np.full((3,), 42, dtype=dtype)
    )
    expected_pad = {"int32": jnp.iinfo(np.int32).min,
                    "float32": np.nan}[dtype_name]
    self.assertAllClose(res,
                        np.array([42, 42, 42, expected_pad, expected_pad, expected_pad],
                                 dtype=dtype))

  def test_block_spec_mapped_dimension(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        in_specs=[
            pl.BlockSpec((None, 4), lambda _: (0, 0)),
            pl.BlockSpec((None, 4), lambda _: (1, 0)),
        ],
        grid=1,
    )
    def add_vectors(x_ref, y_ref, o_ref):
      o_ref[:] = x_ref[:] + y_ref[:]
    xy = jnp.arange(8., dtype=np.float32).reshape((2, 4))
    out = add_vectors(xy, xy)
    out_ref = xy[0] + xy[1]
    np.testing.assert_allclose(out, out_ref)

  @jtu.parameterized_filterable(
      kwargs=[
          dict(shape=(), block_shape=()),
          dict(shape=(2,), block_shape=(2,)),
          dict(shape=(2, 2), block_shape=(2, 2)),
          dict(shape=(3, 3), block_shape=(3, 3)),
          dict(shape=(4, 2), block_shape=(2, 2)),
          dict(shape=(6, 2, 2), block_shape=(2, 2, 2)),
          dict(shape=(6, 2, 2), block_shape=(3, 2, 2)),
          dict(shape=(16, 128), block_shape=(8, 128)),
          dict(shape=(6, 16, 128), block_shape=(2, 8, 128)),
          dict(shape=(6, 16, 128), block_shape=(3, 8, 128)),
          dict(shape=(16, 64), block_shape=(8, 64)),
          dict(shape=(16, 128), block_shape=(4, 128)),
          dict(shape=(16, 128), block_shape=(2, 128)),
          dict(shape=(16, 128), block_shape=(8, 64)),
          # Blocks larger than the number of lands and sublanes.
          dict(shape=(9, 128), block_shape=(9, 64)),
          dict(shape=(9, 128), block_shape=(9, 128)),
          dict(shape=(18, 128), block_shape=(9, 128)),
          dict(shape=(8, 129), block_shape=(8, 129)),
          dict(shape=(9, 129), block_shape=(8, 129)),
          dict(shape=(9, 129), block_shape=(9, 129)),
          # Tiling of small arrays
          dict(shape=(1, 128), block_shape=(4, 128)),
          dict(shape=(2, 128), block_shape=(4, 128)),
          dict(shape=(3, 128), block_shape=(4, 128)),
          dict(shape=(5, 128), block_shape=(8, 128)),
      ]
  )
  def test_block_spec_valid_block_shapes(self, *, shape, block_shape):
    def copy_kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    assert len(shape) == len(block_shape)
    grid = [(sd + bd - 1) // bd for sd, bd in zip(shape, block_shape)]
    x = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)

    test_context = contextlib.nullcontext()
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      if len(block_shape) < 2:
        self.skipTest("TODO(necula): enable this check")
      if len(block_shape) < 2:
        test_context = self.assertRaisesRegex(
            ValueError,
            "TPU lowering currently supports only blocks of rank >= 2")
      else:
        if jaxlib_version < (0, 4, 31):
          evenly_divisible = (
              (block_shape[-1] % 128 == 0
              if shape[-1] >= 128 else block_shape[-1] == shape[-1]) and
              (block_shape[-2] % 8 == 0
              if shape[-2] >= 8 else block_shape[-2] == shape[-2]))
        else:
          evenly_divisible = (
              (block_shape[-1] == shape[-1] or block_shape[-1] % 128 == 0) and
              (block_shape[-2] == shape[-2] or block_shape[-2] % 8 == 0))

        if not evenly_divisible:
          test_context = self.assertRaisesRegex(
              ValueError,
              "last two dimensions of your block shape are divisible by 8 and 128")

    elif jtu.test_device_matches(["gpu"]) and not self.INTERPRET:
      block_size = math.prod(block_shape)
      block_size_is_power_2 = 0 == (block_size & (block_size - 1))
      if not block_size_is_power_2:
        self.skipTest("TODO: Add GPU exception, now RET_CHECK failure")

    with test_context:
      res = self.pallas_call(
          copy_kernel,
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=grid,
          in_specs=[pl.BlockSpec(block_shape, lambda *indices: indices)],
          out_specs=pl.BlockSpec(block_shape, lambda *indices: indices),
      )(x)
      self.assertAllClose(res, x)

  def test_pallas_call_no_grid(self):
    o_ref_shape = None
    def kernel(o_ref):
      nonlocal o_ref_shape
      o_ref_shape = o_ref.shape
      o_ref[...] = jnp.full(o_ref.shape, 42, dtype=np.int32)

    pids = self.pallas_call(kernel,
                            jax.ShapeDtypeStruct((8, 128), dtype=np.int32))()
    self.assertAllClose(pids, np.full((8, 128), 42, dtype=np.int32))
    self.assertEqual(o_ref_shape, (8, 128))

  def test_pallas_call_no_block_spec(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    o_ref_shape = None
    def kernel(o_ref):
      nonlocal o_ref_shape
      o_ref_shape = o_ref.shape
      o_ref[...] = jnp.full(o_ref.shape, pl.program_id(0))

    pids = self.pallas_call(kernel,
                            jax.ShapeDtypeStruct((8,), dtype=np.int32),
                            grid=(1,))()
    self.assertEqual(o_ref_shape, (8,))
    self.assertAllClose(pids, np.array([0] * 8, dtype=np.int32))

  def test_block_spec_no_block_shape_and_no_index_map(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    o_ref_shape = None
    def kernel(o_ref):
      nonlocal o_ref_shape
      o_ref_shape = o_ref.shape
      o_ref[...] = jnp.full(o_ref.shape, pl.program_id(0))

    pids = self.pallas_call(kernel,
                            jax.ShapeDtypeStruct((8,), dtype=np.int32),
                            out_specs=pl.BlockSpec(),
                            grid=(1,))()
    self.assertEqual(o_ref_shape, (8,))
    self.assertAllClose(pids, np.array([0] * 8, dtype=np.int32))

  def test_block_spec_no_block_shape(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    o_ref_shape = None
    def kernel(o_ref):
      nonlocal o_ref_shape
      o_ref_shape = o_ref.shape
      o_ref[...] = jnp.full(o_ref.shape, pl.program_id(0))

    pids = self.pallas_call(kernel,
                            jax.ShapeDtypeStruct((8,), dtype=np.int32),
                            out_specs=pl.BlockSpec(None, lambda i: i),
                            grid=(1,))()
    self.assertEqual(o_ref_shape, (8,))
    self.assertAllClose(pids, np.array([0] * 8, dtype=np.int32))

  def test_block_spec_no_index_map(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    o_ref_shape = None
    def kernel(o_ref):
      nonlocal o_ref_shape
      o_ref_shape = o_ref.shape
      o_ref[...] = jnp.full(o_ref.shape, pl.program_id(0))

    pids = self.pallas_call(kernel,
                            jax.ShapeDtypeStruct((8,), dtype=np.int32),
                            out_specs=pl.BlockSpec((4,)),
                            grid=(1,))()
    self.assertEqual(o_ref_shape, (4,))
    self.assertAllClose(pids[0:4], np.array([0] * 4, dtype=np.int32))

  def test_hoisted_consts(self):
    # See https://github.com/google/jax/issues/21557.
    # to_store will be hoisted as a constant. Choose distinct shapes from in/outs.
    to_store = np.arange(128, dtype=np.float32).reshape((1, 128))
    x = np.arange(16 * 128, dtype=np.float32).reshape((16, 128))

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((64, 128), x.dtype),
        grid=(2,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((32, 128), lambda i: (i, 0)),
    )
    def kernel(src, dst):
      dst[0:1] = to_store

    res = kernel(x)
    self.assertAllClose(res[0:1], to_store)

  def test_vector_slicing(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), floatx),
        grid=1)
    def index(x_ref, idx_ref, o_ref):
      idx = idx_ref[()]
      o_ref[:] = x_ref[idx]

    x = jnp.arange(5.)
    for i in range(4):
      idx = jnp.arange(i, i + 2)
      np.testing.assert_allclose(index(x, idx), x[idx])

  @parameterized.named_parameters(*[
    (f"m_{m}_n_{n}_k_{k}_dtype_{dtype}_bm_{block_size_m}_"
     f"bn_{block_size_n}_bk_{block_size_k}_gm_{group_size_m}", m, n, k, dtype,
     block_size_m, block_size_n, block_size_k, group_size_m)
      for m in [512, 1024]
      for k in [512]
      for n in [512, 1024]
      for dtype in ["float32", "float16"]
      for block_size_m in [64, 128]
      for block_size_n in [64, 128]
      for block_size_k in [32]
      for group_size_m in [8]
      if block_size_m <= m and block_size_n <= n and block_size_k <= k
    ])
  def test_matmul(self, m, n, k, dtype, bm, bn, bk, gm):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    k1, k2 = random.split(random.key(0))
    x = random.normal(k1, (m, k), dtype=dtype)
    y = random.normal(k2, (k, n), dtype=dtype)
    out, expected = matmul(x, y, bm=bm, bn=bn, bk=bk, gm=gm,
                           interpret=self.INTERPRET), jnp.matmul(x, y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.named_parameters(*[
    (f"m_{m}_n_{n}_k_{k}_dtype_{dtype}_bm_{block_size_m}_"
     f"bn_{block_size_n}_bk_{block_size_k}", m, n, k, dtype,
     block_size_m, block_size_n, block_size_k)
      for m in [512, 1024]
      for k in [512]
      for n in [512, 1024]
      for dtype in ["float32", "float16"]
      for block_size_m in [64, 128]
      for block_size_n in [64, 128]
      for block_size_k in [32]
      if block_size_m <= m and block_size_n <= n and block_size_k <= k
    ])
  def test_matmul_block_spec(self, m, n, k, dtype, bm, bn, bk):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    k1, k2 = random.split(random.key(0))
    x = random.normal(k1, (m, k), dtype=dtype)
    y = random.normal(k2, (k, n), dtype=dtype)
    out, expected = matmul_block_spec(x, y, bm=bm, bn=bn, bk=bk,
                                      interpret=self.INTERPRET), jnp.matmul(x, y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.named_parameters(*(
      dict(testcase_name=f"{batch_size}_{size}_{block_size}_{dtype}",
           batch_size=batch_size, size=size, block_size=block_size, dtype=dtype)
      for batch_size in [1, 2, 4, 23]
      for size in [1, 2, 129, 255, 256]
      for block_size in [1, 2, 32, 64, 128, 256]
      for dtype in ["float32"]
      if size < block_size
  ))
  def test_softmax(self, batch_size, size, block_size, dtype):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    @functools.partial(self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((batch_size, size), dtype),
        grid=batch_size)
    def softmax(x_ref, o_ref):
      row_idx = pl.program_id(0)
      x_idx = jnp.arange(block_size)
      row_idxs = (row_idx, x_idx)
      mask = x_idx < x_ref.shape[1]
      row = pl.load(x_ref, row_idxs, mask=mask, other=-float("inf"))
      row_minus_max = row - jnp.max(row, axis=0)
      numerator = jnp.exp(row_minus_max)
      denominator = jnp.sum(numerator, axis=0)
      softmax_output = numerator / denominator
      pl.store(o_ref, row_idxs, softmax_output, mask=mask)

    key = random.key(0)
    x = random.normal(key, [batch_size, size], dtype=dtype)
    np.testing.assert_allclose(softmax(x), jax.nn.softmax(x, axis=-1),
        atol=1e-5, rtol=1e-5)

  def test_unused_ref(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    m, n = 16, 32
    @functools.partial(
        self.pallas_call,
        out_shape=(
          jax.ShapeDtypeStruct((m, n), jnp.float32)
          ), grid=1)
    def dummy(_, o_ref):
      pl.store(o_ref, (jnp.arange(m)[:, None], jnp.arange(n)[None, :]),
               jnp.ones_like(o_ref))

    key = random.key(0)
    x = random.normal(key, (m, n))
    np.testing.assert_allclose(dummy(x), jnp.ones_like(x), atol=1e-5, rtol=1e-5)

  def test_with_input_output_aliasing(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    def add_inplace_kernel(_, o_ref, *, block_size):
      pid = pl.program_id(axis=0)  # we use a 1d launch grid so axis is 0
      block_start = pid * block_size
      offsets = block_start + jnp.arange(block_size, dtype=jnp.int32)
      mask = offsets < o_ref.shape[0]
      x = pl.load(o_ref, (offsets,), mask=mask)
      output = x + 1
      pl.store(o_ref, (offsets,), output, mask=mask)

    grid = (8,)
    size = 8
    dtype = "float32"
    k1 = random.key(0)
    block_size = 1
    x = random.normal(k1, [size], dtype=dtype)
    kernel = functools.partial(add_inplace_kernel, block_size=block_size)
    out = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid, input_output_aliases={0: 0})(x)
    expected = x + 1
    np.testing.assert_allclose(out, expected)

  def test_using_pallas_slice(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    m, n = 32, 4
    out_shape = jax.ShapeDtypeStruct((4, n), floatx)
    @functools.partial(
        self.pallas_call,
        out_shape=out_shape,
        grid=1)
    def slice_kernel(x_ref, y_ref):
      x = pl.load(x_ref, (pl.dslice(0, 4), pl.dslice(0, 4)))
      pl.store(y_ref, (pl.dslice(4), pl.dslice(4)), x)
    x = random.normal(random.key(0), (m, n))
    y = slice_kernel(x)
    y_ref = x[:4]
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)

  def test_pallas_trace_cache(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    trace_count = 0
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def add_one(x_ref, o_ref):
      nonlocal trace_count
      o_ref[()] = x_ref[()] + 1.
      trace_count += 1

    @jax.jit
    def f(x):
      return add_one(add_one(x))

    x = jnp.array(0., dtype=jnp.float32)
    self.assertEqual(f(x), 2.)
    self.assertEqual(trace_count, 1)


class PallasCallInterpreterTest(PallasCallTest):
  INTERPRET = True


class ApiErrorTest(PallasBaseTest):
  def test_pallas_kernel_args_mismatch(self):
    a = np.arange(256, dtype=np.int32)
    f = self.pallas_call(lambda x_ref: None,  # Missing o_ref
                         out_shape=a)
    with self.assertRaisesRegex(
        TypeError,
        "takes 1 positional argument but 2 were given"):
      f(a)

  @parameterized.named_parameters(
      ("array", 0),
      ("empty_tuple", ())
  )
  def test_pallas_call_error_kernel_returns_something(self, returns):
    a = np.arange(256, dtype=np.int32)
    # The kernel should not return anything
    f = self.pallas_call(lambda x_ref, o1_ref, o2_ref: returns,
                         out_shape=(a, a))
    with self.assertRaisesRegex(
        ValueError,
        "The kernel function in a pallas_call should return None"):
      f(a)

  def test_pallas_call_in_specs_not_a_sequence(self):
    a = np.arange(256, dtype=np.int32)
    with self.assertRaisesRegex(
        ValueError,
        "`in_specs` must be a tuple or a list"):
      _ = self.pallas_call(lambda x_ref, o1_ref: None,
                           out_shape=a,
                           in_specs=pl.BlockSpec((4,), lambda: 0))

  def test_pallas_call_in_specs_mismatch_inputs(self):
    a = np.arange(256, dtype=np.int32)
    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=a,
                         in_specs=[pl.BlockSpec((4,), lambda: 0),
                                   pl.BlockSpec((4,), lambda: 0)])
    with self.assertRaisesRegex(
        ValueError,
        re.compile("Pytree for `in_specs` and inputs do not match. "
                   "There are 1 mismatches, including:"
                   ".* at \\[1\\], `in_specs` is a pytree leaf but "
                   "inputs is a.*", re.DOTALL)):
      f(a, dict(a=a))

  def test_pallas_call_index_map_wrong_number_of_arguments(self):
    a = np.arange(256, dtype=np.int32)
    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=a,
                         in_specs=[pl.BlockSpec((4,), lambda i, j: 0)])
    with self.assertRaisesRegex(
        TypeError,
        "missing 2 required positional arguments: 'i' and 'j'"):
      f(a)

  def test_pallas_call_index_map_wrong_number_of_results(self):
    a = np.arange(256, dtype=np.int32)
    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=a,
                         in_specs=[pl.BlockSpec((4,), lambda: (0, 0))])
    with self.assertRaisesRegex(
        ValueError,
        "Index map for inputs\\[0\\] must return 1 values to match .*Currently returning 2 values."):
      f(a)

  def test_pallas_call_index_map_captures_consts(self):
    a = np.arange(256, dtype=np.int32)
    index_map_result = np.array([0], dtype=np.int32)
    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=a,
                         in_specs=[pl.BlockSpec((4,), lambda: index_map_result)])
    with self.assertRaisesRegex(
        NotImplementedError,
        "Index map for inputs\\[0\\] captures constants"):
      f(a)

  def test_pallas_call_out_specs_mismatch_shape(self):
    a = np.arange(256, dtype=np.int32)
    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=[a, a],
                         out_specs=[pl.BlockSpec((6,), lambda i: i)])
    with self.assertRaisesRegex(
        ValueError,
        re.compile("Pytree for `out_specs` and `out_shape` do not match. There are 1 mismatches, including:"
         ".* `out_specs` is a tuple of length 1 but `out_shape` is a tuple of length 2.*", re.DOTALL)):
      f(a)


  def test_pallas_call_block_shape_ndim_mismatch(self):
    a = np.arange(256, dtype=np.int32)
    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=[a],
                         in_specs=[pl.BlockSpec((1, 1), lambda: (0, 0))])
    with self.assertRaisesRegex(
        ValueError,
        "Block shape for inputs\\[0\\] .* must have the same number of dimensions as the "
        "array shape"):

      f(a)

    f = self.pallas_call(lambda x_ref, o1_ref: None,
                         out_shape=[a],
                         out_specs=[pl.BlockSpec((1, 1), lambda: 0)])
    with self.assertRaisesRegex(
        ValueError,
        "Block shape for outputs\\[0\\] .* must have the same number of dimensions as the "
        "array shape"):
      f(a)

  def test_pallas_call_input_output_aliases_errors(self):
    x = np.arange(8 * 128, dtype=np.int32).reshape((8, 128))

    with self.assertRaisesRegex(
        ValueError,
        "input_output_aliases contains the mapping '2:0' with input index 2 "
        "outside the range .*"):
      self.pallas_call(lambda x_ref, y_ref, o1_ref: None,
                       out_shape=[x],
                       input_output_aliases={2: 0})(x, x)

    with self.assertRaisesRegex(
        ValueError,
        "input_output_aliases contains the mapping '1:1' with output index 1 "
        "outside the range .*"):
      self.pallas_call(lambda x_ref, y_ref, o1_ref: None,
                       out_shape=[x],
                       input_output_aliases={1: 1})(x, x)

    y = np.concatenate([x, x], axis=0)
    with self.assertRaisesRegex(
        ValueError,
        "input_output_aliases contains the mapping '1:0' referring to "
        "input\\[1\\] with abstract value .*int32\\[16,128\\].* "
        "output\\[0\\] with a different abstract value .*int32\\[8,128\\]"):
      self.pallas_call(lambda x_ref, y_ref, o1_ref: None,
                       out_shape=[x],
                       input_output_aliases={1: 0})(x, y)

    with self.assertRaisesRegex(
        ValueError,
        "input_output_aliases contains the mapping '1:0' referring to "
        "input\\[1\\] with abstract value .*int32\\[8,128\\].* "
        "output\\[0\\] with a different abstract value .*float32\\[8,128\\]"):
      self.pallas_call(lambda x_ref, y_ref, o1_ref: None,
                       out_shape=[jax.ShapeDtypeStruct(x.shape, jnp.float32)],
                       input_output_aliases={1: 0})(x, x)


class ApiErrorInterpreterTest(ApiErrorTest):
  INTERPRET = True


class PallasCallInputOutputAliasingTest(PallasBaseTest):

  def test_basic_input_output_aliasing(self):
    # Input needs to be big so it doesn't fit in VMEM
    size = 1024
    if jtu.is_device_cuda():
      # Reduce the size on CUDA to avoid OOM.
      size = 256
    x = jnp.ones((32, size, size))
    expected = x + 1

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...] + 1.
    @functools.partial(jax.jit, donate_argnums=(0,))
    def f(x):
      return self.pallas_call(
          kernel,
          out_shape=x,
          in_specs=[pl.BlockSpec((None, size, size), lambda i: (i, 0, 0))],
          out_specs=pl.BlockSpec((None, size, size), lambda i: (i, 0, 0)),
          grid=(x.shape[0],),
          input_output_aliases={0: 0},
      )(x)
    o = f(x)
    np.testing.assert_array_equal(o, expected)
    compiled = f.lower(jax.ShapeDtypeStruct(x.shape, x.dtype)).compile()
    mem_analysis = compiled.memory_analysis()
    expected_num_bytes = np.prod(x.shape) * x.dtype.itemsize
    self.assertEqual(mem_analysis.alias_size_in_bytes, expected_num_bytes)
    self.assertEqual(mem_analysis.temp_size_in_bytes, 0)


class PallasCallInputOutputAliasingInterpreterTest(PallasBaseTest):
  INTERPRET = True


class PallasControlFlowTest(PallasBaseTest):

  def setUp(self):
    super().setUp()
    if self.INTERPRET:
      self.skipTest("Control flow not supported in interpreter mode yet.")

  def test_loop_with_float64_carry(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    # Test that the jnp.zeros(f64) loop init_val is actually f64, and that
    # fori_loop handles i64 index variables, i.e. error: 'scf.for' op  along
    # control flow edge from Region #0 to Region #0: source type #0
    # 'tensor<4xf64>' should match input type #0 'tensor<4xf32>'
    with config.enable_x64(True):
      @functools.partial(self.pallas_call,
                         out_shape=jax.ShapeDtypeStruct((4,), jnp.float64),
                         grid=1,
                     )
      def f(x_ref, y_ref):
        def body(i, acc):
          # TODO(sharadmv): DCE loop index but retain carry breaks scan pattern.
          # return acc + x_ref[...]
          return acc + x_ref[...] + i * 0
        y_ref[...] = lax.fori_loop(
            0, 3, body, jnp.zeros((4,), jnp.float64))

      np.testing.assert_allclose(np.arange(1, 5.) * 3,
                                 f(jnp.arange(1, 5., dtype=jnp.float64)))

  def test_cond_simple(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    arg = jnp.float32(0.)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                   )
    def f(branch_ref, x_ref, y_ref):
      y_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x: x**2, lambda x: -x),
          x_ref[...])
    y = f(jnp.int32(0), arg + 3.)
    self.assertEqual(y, 9.)
    y = f(jnp.int32(1), arg + 2.)
    self.assertEqual(y, -2.)

  def test_cond_threebranch(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    arg = jnp.float32(0.)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       grid=1,
                   )
    def f(branch_ref, x_ref, y_ref):
      y_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x: x**2, lambda x: -x, lambda x: -x**2),
          x_ref[...])
    y = f(jnp.int32(0), arg + 3.)
    self.assertEqual(y, 9.)
    y = f(jnp.int32(1), arg + 2.)
    self.assertEqual(y, -2.)
    y = f(jnp.int32(2), arg + 4.)
    self.assertEqual(y, -16.)

  @parameterized.parameters(1, 2, 4, 8)
  def test_cond_vectors(self, block_size):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")
    arg = jnp.float32([0.] * 8)
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
        in_specs=[
            pl.BlockSpec((), lambda _: ()),
            pl.BlockSpec((block_size,), lambda i: i),
        ],
        out_specs=pl.BlockSpec((block_size,), lambda i: i),
        grid=pl.cdiv(arg.shape[0], block_size),
    )
    def f(branch_ref, x_ref, y_ref):
      y_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x: x**2, lambda x: -x),
          x_ref[...])
    y = f(jnp.int32(0), arg + 3.)
    np.testing.assert_allclose(y, arg + 9.)
    y = f(jnp.int32(1), arg + 2.)
    np.testing.assert_allclose(y, arg - 2.)

  @parameterized.parameters(1, 2, 4, 8)
  def test_cond_threebranch_vectors(self, block_size):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")
    arg = jnp.float32([0.] * 8)
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
        in_specs=[
            pl.BlockSpec((), lambda _: ()),
            pl.BlockSpec((block_size,), lambda i: i),
        ],
        out_specs=pl.BlockSpec((block_size,), lambda i: i),
        grid=pl.cdiv(arg.shape[0], block_size),
    )
    def f(branch_ref, x_ref, y_ref):
      y_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x: x**2, lambda x: -x, lambda x: -x**2),
          x_ref[...])
    y = f(jnp.int32(0), arg + 3.)
    np.testing.assert_allclose(y, arg + 9.)
    y = f(jnp.int32(1), arg + 2.)
    np.testing.assert_allclose(y, arg - 2.)
    y = f(jnp.int32(2), arg + 4.)
    np.testing.assert_allclose(y, arg - 16.)

  @parameterized.parameters(*itertools.product([1, 8], [1, 2, 4]))
  def test_cond_threebranch_matrix_out(self, bx, by):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")
    x = jnp.arange(64.)[:, None]
    y = jnp.arange(128.0)[None, :]

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), floatx),
        in_specs=[
            pl.BlockSpec((), lambda _, __: ()),
            pl.BlockSpec((bx, 1), lambda i, _: (i, 0)),
            pl.BlockSpec((1, by), lambda _, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bx, by), lambda i, j: (i, j)),
        grid=(pl.cdiv(x.shape[0], bx), pl.cdiv(y.shape[1], by)),
    )
    def f(branch_ref, x_ref, y_ref, o_ref):
      o_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x, y: (x - y)**2,
           lambda x, y: -jnp.abs(x - y),
           lambda x, y: jnp.sqrt(jnp.abs(x - y))),
          x_ref[...],
          y_ref[...])
    np.testing.assert_allclose(f(jnp.int32(0), x, y), (x - y)**2)
    np.testing.assert_allclose(f(jnp.int32(1), x, y), -jnp.abs(x - y))
    np.testing.assert_allclose(f(jnp.int32(2), x, y), jnp.sqrt(jnp.abs(x - y)))

  def test_nested_conds(self):
    def kernel(y_ref):
      def select(pred, x, y, nesting=0):
        def _true():
          if nesting == 0:
            return x + 1
          return select(x == nesting, x, y, nesting=nesting - 1)

        def _false():
          if nesting == 0:
            return y + 1
          return select(y == nesting, x, y, nesting=nesting - 1)

        return jax.lax.cond(pred, _true, _false)

      j = pl.program_id(0)
      j = select(j == 0, j, j, nesting=4)
      y_ref[...] = j * jnp.ones_like(y_ref)

    pl.pallas_call(
        kernel,
        grid=(1,),
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
    )()
    return

  def test_conditional_write(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    arg = jnp.arange(8, dtype=jnp.float32)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                   )
    def f(branch_ref, x_ref, out_ref):
      out_ref[...] = -x_ref[...]
      def if_true(z):
        out_ref[4] = z
        return ()
      jax.lax.cond(branch_ref[...], if_true, lambda z: (), x_ref[6])
    np.testing.assert_allclose(f(jnp.bool_(True), arg),
                               jnp.float32([0., -1, -2, -3, 6, -5, -6, -7]))
    np.testing.assert_allclose(f(jnp.bool_(False), arg),
                               -arg)

    # We actually expect the assertion failure in linearize, but this also
    # covers another case where an effect was causing an earlier assertion
    # failure.
    with self.assertRaises(AssertionError):
      # Notably, we should not have a ValueError for mismatched Read<N> effect.
      _ = jax.grad(lambda x: jnp.sum(f(jnp.bool_(True), x)**2))(arg)
      # np.testing.assert_allclose(
      #     dx, jnp.float32([0., 2, 4, 6, 0, 10, 12 + 12, 14]))

  def test_scan_cond_vm_explicit_ref_arg(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    program = jnp.int32([0, 1, 2, 3, 2])
    params = jnp.arange(len(program) * 3., dtype=jnp.float32)
    params = params.reshape(len(program), 3)
    x = jnp.arange(7., dtype=jnp.float32)
    bx = 4

    @jax.jit
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
        in_specs=[
            pl.BlockSpec(program.shape, lambda _: (0,)),  # program
            pl.BlockSpec(params.shape, lambda _: (0, 0)),  # params
            pl.BlockSpec((bx,), lambda i: (i,)),
        ],  # x
        out_specs=pl.BlockSpec((bx,), lambda i: (i,)),
        grid=pl.cdiv(x.shape[0], bx),
    )
    def f(program_ref, params_ref, x_ref, out_ref):
      x = x_ref[...]

      def body_fn(i, args):
        state, program_ref, params_ref = args
        opcode = program_ref[i]
        state = jax.lax.switch(
            opcode,
            (lambda state, params, i: state + params[i, 0] * 2.**i * x,
             lambda state, params, i: state + params[i, 1] * 2.**i * x,
             lambda state, params, i: state + params[i, 2] * 2.**i * x,
             lambda state, params, i: state + params[i, 1] * 2.**i * x,
             ),
            state, params_ref, i)
        return state, program_ref, params_ref
      out_ref[...] = jax.lax.fori_loop(
          0, len(program), body_fn,
          (jnp.zeros(x.shape, dtype=jnp.float32), program_ref, params_ref))[0]

    expected = (x * params[0, 0] +
                2 * x * params[1, 1] +
                4 * x * params[2, 2] +
                8 * x * params[3, 1] +
                16 * x * params[4, 2])
    np.testing.assert_allclose(f(program, params, x), expected)

    with self.assertRaises(AssertionError):
      jax.value_and_grad(lambda params, x: f(program, params, x).sum())(
          params, x)

  def test_scan_cond_vm_closing_over_ref(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    # ** Difference is the closure over params_ref in the switch branches. **
    program = jnp.int32([0, 1, 2, 3, 2, -1])
    params = jnp.arange(len(program) * 3., dtype=jnp.float32)
    params = params.reshape(len(program), 3)
    x = jnp.arange(7., dtype=jnp.float32)
    bx = 4

    @jax.jit
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
        in_specs=[
            pl.BlockSpec(program.shape, lambda _: (0,)),  # program
            pl.BlockSpec(params.shape, lambda _: (0, 0)),  # params
            pl.BlockSpec((bx,), lambda i: (i,)),
        ],  # x
        out_specs=pl.BlockSpec((bx,), lambda i: (i,)),
        grid=pl.cdiv(x.shape[0], bx),
    )
    def f(program_ref, params_ref, x_ref, out_ref):
      x = x_ref[...]

      def body_fn(i, args):
        state, program_ref, params_ref = args
        opcode = program_ref[i] + 1
        state = jax.lax.switch(
            opcode,
            (lambda state, *_: state,
             lambda state, i: state + params_ref[i, 0] * 2.**i * x,
             lambda state, i: state + params_ref[i, 1] * 2.**i * x,
             lambda state, i: state + params_ref[i, 2] * 2.**i * x,
             lambda state, i: state + params_ref[i, 1] * 2.**i * x,
             ),
            state, i)
        return state, program_ref, params_ref
      out_ref[...] = jax.lax.fori_loop(
          0, len(program), body_fn,
          (jnp.zeros(x.shape, dtype=jnp.float32), program_ref, params_ref))[0]

    expected = (x * params[0, 0] +
                2 * x * params[1, 1] +
                4 * x * params[2, 2] +
                8 * x * params[3, 1] +
                16 * x * params[4, 2])
    np.testing.assert_allclose(f(program, params, x), expected)

    with self.assertRaises(AssertionError):
      jax.value_and_grad(lambda params, x: f(program, params, x).sum())(
          params, x)

  def test_fori_loop_simple(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      y_ref[...] = x_ref[...]
      def body(i, _):
        y_ref[...] += 1
      lax.fori_loop(0, 5, body, None)
    y = f(0)
    self.assertEqual(y, 5)

  def test_fori_loop_with_nonzero_lower_bound(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      y_ref[...] = x_ref[...]
      def body(i, _):
        y_ref[...] += i
      lax.fori_loop(2, 5, body, None)
    y = f(6)
    self.assertEqual(y, 6 + 2 + 3 + 4)

  def test_fori_loop_accumulates(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      def body(i, acc):
        return acc + 1
      acc = lax.fori_loop(0, 5, body, 0)
      y_ref[...] = acc
    y = f(0)
    self.assertEqual(y, 5)

  def test_fori_loop_accumulates_with_index(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      def body(i, acc):
        return acc + i
      acc = lax.fori_loop(0, 5, body, 0)
      y_ref[...] = acc
    y = f(0)
    self.assertEqual(y, 10)

  def test_fori_loop_with_writing_to_index(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((8,), jnp.int32))
    def f(y_ref):
      def body(i, _):
        y_ref[i] = i
      lax.fori_loop(0, y_ref.shape[0], body, None)
    y = f()
    np.testing.assert_allclose(y, jnp.arange(8))

  def test_fori_loop_with_dynamic_indices(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(lb_ref, ub_ref, y_ref):
      y_ref[...] = 0
      def body(i, a):
        y_ref[...] += i
        return a
      lax.fori_loop(lb_ref[...], ub_ref[...], body, 1)
    y = f(2, 5)
    np.testing.assert_allclose(y, 2 + 3 + 4)
    y = f(1, 8)
    np.testing.assert_allclose(y, sum(range(1, 8)))

  def test_simple_while(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      x = x_ref[...]
      y_ref[...] = 0
      def cond(x):
        return x < 5
      def body(x):
        y_ref[...] += 1
        return x + 1
      lax.while_loop(cond, body, x)
    y = f(0)
    self.assertEqual(y, 5)

  def test_simple_while_with_only_values(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(y_ref):
      def cond(acc):
        return acc < 5
      def body(acc):
        acc += 1
        return acc
      acc = lax.while_loop(cond, body, 0)
      y_ref[...] = acc
    y = f()
    self.assertEqual(y, 5)

  def test_while_with_dynamic_condition(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(i_ref, y_ref):
      y_ref[...] = 0
      n_iter = i_ref[...]
      def cond(i):
        return i < n_iter
      def body(i):
        y_ref[...] += 1
        return i + 1
      _ = lax.while_loop(cond, body, 0)

    self.assertEqual(f(1), 1)
    self.assertEqual(f(4), 4)
    self.assertEqual(f(100), 100)

  def test_vmap_of_while_with_dynamic_condition(self):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("TODO: error on TPU")

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(i_ref, y_ref):
      y_ref[...] = 0
      n_iter = i_ref[...]
      def cond(i):
        return i < n_iter
      def body(i):
        y_ref[...] += 1
        return i + 1
      _ = lax.while_loop(cond, body, 0)

    x = jnp.array([1, 4, 100])
    np.testing.assert_array_equal(jax.vmap(f)(x), x)

  def test_range_while_loop(self):
    """Tests lowering of a while_loop which can reduce to a fori_loop."""

    def kernel(x_ref, r_ref):
      @pl.when(pl.program_id(0) == 0)
      def _():
        pl.store(r_ref, (0, 0), 0)

      def cond(carry):
        i, j = carry
        return i < j

      def body(carry):
        io, j = carry
        i = io - 128
        sl = jax.lax.div(i, 128)
        l = jax.lax.rem(i, 128)
        v = x_ref[0, sl, l]
        s = pl.load(r_ref, (0, 0))
        pl.store(r_ref, (0, 0), s + v)
        return io + 1, j

      i = 128
      j = 128 + 1024
      i, j = jax.lax.while_loop(cond, body, (i, j))

    x = jnp.arange(4096)
    x = jnp.reshape(x, [4, 8, 128])

    r = pl.pallas_call(
        kernel,
        grid=(1,),
        out_specs=pl.BlockSpec((1, 1), memory_space=smem_on_tpu()),
        out_shape=jax.ShapeDtypeStruct([1, 1], intx),
        in_specs=[
            pl.BlockSpec(
                (1, 8, 128),
                lambda i: (i, 0, 0),
                memory_space=smem_on_tpu(),
            )
        ],
    )(x)
    expected = jnp.sum(jnp.arange(1024))
    np.testing.assert_array_equal(r, expected)

  def test_fori(self):
    """Tests lowering of a while_loop which can reduce to a fori_loop."""

    def kernel(lb_ref, ub_ref, o_ref):
      o_ref[0, 0] = 0

      def body(i, _):
        o_ref[0, 0] += 1

      jax.lax.fori_loop(lb_ref[0, 0], ub_ref[0, 0], body, None)

    smem = pl.BlockSpec(memory_space=smem_on_tpu())
    r = pl.pallas_call(
        kernel,
        in_specs=(smem, smem),
        out_specs=smem,
        out_shape=jax.ShapeDtypeStruct([1, 1], jnp.int32),
    )(*(jnp.array([[x]]) for x in (2, 6)))
    np.testing.assert_array_equal(r, 4)

  def test_non_range_while_loop(self):
    """Tests lowering of a while_loop which cannot reduce to a fori_loop."""

    def kernel(x_ref, r_ref):
      @pl.when(pl.program_id(0) == 0)
      def _():
        pl.store(r_ref, (0, 0), 0)

      def cond(state):
        i, s = state
        return jnp.logical_and(i < 1024, s < 1024)

      def body(state):
        i, s = state
        sl = jax.lax.div(i, 128)
        l = jax.lax.rem(i, 128)
        v = pl.load(x_ref, (0, sl, l))
        return i + 1, s + v

      i = jnp.int32(0)
      s = pl.load(r_ref, (0, 0))

      i, s = jax.lax.while_loop(cond, body, (i, s))
      pl.store(r_ref, (0, 0), s)

    x = jnp.arange(4096)
    x = jnp.reshape(x, [4, 8, 128])

    r = pl.pallas_call(
        kernel,
        grid=(4,),
        out_specs=pl.BlockSpec((1, 1), memory_space=smem_on_tpu()),
        out_shape=jax.ShapeDtypeStruct([1, 1], intx),
        in_specs=[
            pl.BlockSpec(
                (1, 8, 128),
                lambda i: (i, 0, 0),
                memory_space=smem_on_tpu(),
            )
        ],
    )(x)
    np.testing.assert_array_equal(r, [[1035]])

  def test_vector_carry_while_loop(self):
    """Tests lowering of a while_loop which carries a vector quantity."""
    if jtu.test_device_matches(["gpu"]) and not self.INTERPRET:
      self.skipTest("TODO: slice not implemented on GPU")
    def kernel(x_ref, r_ref):

      def cond(v):
        return v[0, 0] < 16

      def body(v):
        return v * 2

      r_ref[:] = jax.lax.while_loop(cond, body, x_ref[:])

    x = jnp.full((8, 128), 3, dtype=jnp.int32)
    fn = pl.pallas_call(
        kernel,
        grid=(1,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
    )
    r = fn(x)
    reduced = jnp.sum(r)
    # 3 -> 6 -> 12 -> 24
    np.testing.assert_array_equal(reduced, 1024 * 24)

  @parameterized.named_parameters(
      ('1x128', (1, 128)),
      ('2x128', (2, 128)),
      ('4x128', (4, 128)),
      ('8x128', (8, 128)),
      ('8x256', (8, 256)),
  )
  def test_while_loop_carry_memref(self, shape):
    """Tests a while loop carrying a memref."""

    # TODO(hmckenzie): Investigate further why this occurs.
    if shape == (1, 128):
      self.skipTest('memref<1x128> inexplicably doubles to 2x128.')

    def kernel(out_ref, bound):
      def cond(i):
        return i < bound

      def body(i):
        out_ref[0, i] = 2
        return i + 1

      jax.lax.while_loop(cond, body, 0)

    x = jnp.asarray([1, 1, 1, 1])
    x = jnp.asarray(x)
    x = jnp.pad(x, (0, np.prod(shape) - 4), constant_values=0)
    x = jnp.reshape(x, shape)
    kernel = functools.partial(kernel, bound=x.shape[1])

    fn = pl.pallas_call(
        kernel,
        grid=(1,),
        out_specs=[
            pl.BlockSpec(shape, lambda i: (0, 0), memory_space=smem_on_tpu()),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(shape, jnp.int32),
        ],
    )
    y = fn()[0]
    np.testing.assert_array_equal(y[0, 0], 2)
    np.testing.assert_array_equal(y[0, 1], 2)
    np.testing.assert_array_equal(y[0, 2], 2)
    np.testing.assert_array_equal(y[0, 3], 2)

  def test_nested_while_loop(self):
    """Tests lowering a nested while_loop."""
    if jtu.test_device_matches(["gpu"]) and not self.INTERPRET:
      self.skipTest("TODO: assertion error on GPU")

    def kernel(in_key_ref, out_segment_count, out_size_ref, key_count):
      # Compute the length of contiguous segments of keys.

      def inner_cond(carry):
        i, prev_key = carry
        sl = jax.lax.div(i, 128)
        l = jax.lax.rem(i, 128)
        key = jax.lax.cond(
            i < key_count, lambda i: in_key_ref[sl, l], lambda i: -1, i
        )
        return jnp.logical_and(i < key_count, key == prev_key)

      def inner_body(carry):
        i, key = carry
        return i + 1, key

      def outer_cond(carry):
        i, _ = carry
        return i < key_count

      def outer_body(carry):
        i, next_out_idx = carry
        sl = jax.lax.div(i, 128)
        l = jax.lax.rem(i, 128)
        key = in_key_ref[sl, l]
        end, _ = jax.lax.while_loop(inner_cond, inner_body, (i + 1, key))

        sl = jax.lax.div(next_out_idx, 128)
        l = jax.lax.rem(next_out_idx, 128)
        out_size_ref[sl, l] = end - i
        return end, next_out_idx + 1

      _, count = jax.lax.while_loop(outer_cond, outer_body, (0, 0))
      out_segment_count[0, 0] = count

    keys = [4, 4, 4, 3, 2, 2, 7, 7, 7, 7]
    keys = jnp.asarray(keys)
    real_keys = keys.shape[0]
    key_count = 1024
    keys = jnp.pad(keys, (0, key_count - real_keys), constant_values=32768)
    keys = jnp.reshape(keys, (8, 128))
    kernel_fn = functools.partial(kernel, key_count=key_count)

    fn = pl.pallas_call(
        kernel_fn,
        grid=(1,),
        in_specs=[
            # keys.
            pl.BlockSpec((8, 128), lambda i: (0, 0), memory_space=smem_on_tpu()),
        ],
        out_specs=[
            # Segments found.
            pl.BlockSpec((1, 1), memory_space=smem_on_tpu()),
            # Segment sizes.
            pl.BlockSpec((8, 128), memory_space=smem_on_tpu()),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((1, 1), jnp.int32),
            jax.ShapeDtypeStruct((8, 128), jnp.int32),
        ],
    )
    count, sizes = fn(keys)
    np.testing.assert_equal(count[0, 0], jnp.asarray(5))
    np.testing.assert_equal(sizes[0, 0], jnp.asarray(3))
    np.testing.assert_equal(sizes[0, 1], jnp.asarray(1))
    np.testing.assert_equal(sizes[0, 2], jnp.asarray(2))
    np.testing.assert_equal(sizes[0, 3], jnp.asarray(4))
    np.testing.assert_equal(sizes[0, 4], jnp.asarray(key_count - real_keys))


class PallasControlFlowInterpreterTest(PallasControlFlowTest):
  INTERPRET = True

AD_TEST_CASES = [
    ("square", lambda x: x * x),
    ("square_pow", lambda x: x ** 2),
    ("square_fn", jnp.square),
    ("add_one", lambda x: x + 1.),
    ("exp", jnp.exp),
    ("reciprocal", jnp.reciprocal),
    ("one_over_x", lambda x: 1. / x),
    ("recip_exp_sq", lambda x: jnp.reciprocal(jnp.exp(x) ** 2)),
    ("exp_neg_sq", lambda x: jnp.exp(-x) ** 2),
    ("sin", jnp.sin),
    ("tanh", jnp.tanh),
]


class PallasCallAutodifferentiationTest(PallasBaseTest):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["tpu"]):
      # TODO: most tests fail on TPU in non-interpreter mode
      self.skipTest("On TPU the test works only in interpret mode")
    # TODO: improve tolerance setting
    self.tol = 1e-5
    self.grad_tol = jtu.default_gradient_tolerance[np.dtype(jnp.float32)]

  @parameterized.named_parameters(*AD_TEST_CASES)
  def test_jvp(self, impl):
    grad_tol = self.grad_tol
    if jtu.test_device_matches(["tpu"]) and "recip_exp_sq" in self._testMethodName:
      grad_tol = 1e-1

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), floatx),
        grid=1)
    def pallas_impl(x_ref, o_ref):
      x = x_ref[()]
      o_ref[()] = impl(x)

    k1, k2 = random.split(random.key(0))
    x = random.normal(k1)
    t = random.normal(k2)
    out_primal, out_tangent = jax.jvp(pallas_impl, (x,), (t,))
    out_primal_ref, out_tangent_ref = jax.jvp(impl, (x,), (t,))
    np.testing.assert_allclose(out_primal, out_primal_ref, atol=self.tol,
                               rtol=self.tol)
    np.testing.assert_allclose(out_tangent, out_tangent_ref, atol=self.tol,
                               rtol=self.tol)
    jtu.check_grads(pallas_impl, (x,), modes=["fwd"], order=2,
                    atol=grad_tol, rtol=grad_tol)

  @parameterized.named_parameters(*AD_TEST_CASES)
  def test_pallas_around_grad(self, impl):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), floatx),
        name=self.id().split(".")[-1],
        grid=1)
    def pallas_impl(x_ref, o_ref):
      x = x_ref[()]
      o_ref[()] = jax.grad(impl)(x)

    x = random.normal(random.key(0))
    out_grad = pallas_impl(x)
    out_grad_ref = jax.grad(impl)(x)
    np.testing.assert_allclose(out_grad, out_grad_ref, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(*AD_TEST_CASES)
  def test_jvp_slice(self, impl):
    grad_tol = self.grad_tol
    if jtu.test_device_matches(["tpu"]) and "tanh" in self._testMethodName:
      grad_tol = 1e-1

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), floatx),
        grid=1)
    def pallas_impl(x_ref, o_ref):
      x = x_ref[jnp.arange(2)]
      o_ref[jnp.arange(2)] = jnp.zeros(2)
      o_ref[2 + jnp.arange(2)] = impl(x)

    k1, k2 = random.split(random.key(0))
    x = random.normal(k1, (8,))
    t = random.normal(k2, (8,))
    out_primal, out_tangent = jax.jvp(pallas_impl, (x,), (t,))
    out_primal_ref, out_tangent_ref = jax.jvp(
        lambda x: jnp.concatenate([jnp.zeros(2), impl(x[:2])]), (x,), (t,))
    np.testing.assert_allclose(out_primal, out_primal_ref, atol=self.tol,
                               rtol=self.tol)
    np.testing.assert_allclose(out_tangent, out_tangent_ref, atol=self.tol,
                               rtol=self.tol)
    jtu.check_grads(pallas_impl, (x,), modes=["fwd"], order=2,
                    atol=grad_tol, rtol=grad_tol)

  def test_custom_jvp_call(self):
    @functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
    def softmax(x, axis=-1):
      unnormalized = jnp.exp(x - jnp.max(x, axis, keepdims=True))
      return unnormalized / jnp.sum(unnormalized, axis, keepdims=True)

    @softmax.defjvp
    def softmax_jvp(axis, primals, tangents):
      (x,), (x_dot,) = primals, tangents
      y = softmax(x, axis)
      return y, y * (x_dot - (y * x_dot).sum(axis, keepdims=True))

    m, n = 16, 32
    x = random.normal(random.key(0), (m, n))

    @functools.partial(self.pallas_call, out_shape=x, grid=1)
    def softmax_kernel(x_ref, y_ref):
      y_ref[:] = softmax(x_ref[:])

    np.testing.assert_allclose(softmax_kernel(x), jax.nn.softmax(x), atol=1e-7)

  # TODO(sharadmv): enable this when we update Triton
  # def test_jvp_matmul(self):
  #   k1, k2 = random.split(random.key(0))
  #   x = random.normal(k1, (256, 128))
  #   y = random.normal(k2, (128, 64))
  #   bm, bn, bk, gm = 64, 128, 32, 8
  #   mm = functools.partial(matmul, bm=bm, bn=bn, bk=bk, gm=gm,
  #                          interpret=self.INTERPRET)
  #   jtu.check_grads(mm, (x, y), modes=["fwd"], order=1)


class PallasCallAutodifferentiationInterpreterTest(PallasCallAutodifferentiationTest):
  INTERPRET = True


class PallasOutOfBoundsInterpreterTest(PallasBaseTest):
  INTERPRET = True

  def test_interpret_mode_out_of_bounds_access(self):
    block_size = 32
    dtype = jnp.float32
    # Create input tensors which require a reduction along an axis
    # not divisible by block_size.
    x = jax.random.normal(jax.random.key(0),
                          (block_size, block_size + 1),
                          dtype=dtype)
    y = jax.random.normal(jax.random.key(1),
                          (block_size + 1, block_size),
                          dtype=dtype)
    expected = x @ y

    in_specs = [
        pl.BlockSpec((block_size, block_size), lambda i, j, k: (i, k)),
        pl.BlockSpec((block_size, block_size), lambda i, j, k: (k, j)),
    ]
    out_spec = pl.BlockSpec((block_size, block_size), lambda i, j, k: (i, j))

    def _unmasked_matmul_kernel(x_ref, y_ref, o_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        o_ref[...] = jnp.zeros_like(o_ref)

      o_ref[...] += x_ref[...] @ y_ref[...]

    out = self.pallas_call(
        _unmasked_matmul_kernel,
        out_shape=expected,
        grid=(1, 1, 2),
        in_specs=in_specs,
        out_specs=out_spec)(x, y)

    # With a naive matmul implementation, using uninitialized values (NaN) will
    # cause the overall output to be NaN.
    with self.subTest('UnmaskedIsNaN'):
      np.testing.assert_allclose(
          np.isnan(out), jnp.ones_like(out, dtype=jnp.bool_)
      )

    def _masked_matmul_kernel(x_ref, y_ref, o_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        o_ref[:, :] = jnp.zeros_like(o_ref)

      # Create a validity mask for OOB values.
      num_valid = x.shape[1] - pl.program_id(2) * block_size
      num_valid = jnp.minimum(num_valid, block_size)
      mask = jnp.tril(jnp.ones_like(x_ref[:, :]))[num_valid - 1][jnp.newaxis, :]
      mask = jnp.repeat(mask, block_size, axis=0)

      # Mask and multiply.
      masked_x = jnp.where(mask, x_ref[:, :], 0.0)
      masked_y = jnp.where(mask.T, y_ref[:, :], 0.0)
      o_ref[:, :] += masked_x @ masked_y

    out = self.pallas_call(
        _masked_matmul_kernel,
        out_shape=expected,
        grid=(1, 1, 2),
        in_specs=in_specs,
        out_specs=out_spec)(x, y)

    # TODO(justinfu): This test has low precision on GPU. Improve precision.
    if jtu.test_device_matches(["gpu"]):
      atol = 1e-2
    else:
      atol = 1e-5

    # With a masked matmul implementation, uninitialized values will be
    # masked before computation. This should return the correct result.
    with self.subTest('MaskedOutputIsCorrect'):
      np.testing.assert_allclose(out, expected, atol=atol)


class PallasCheckifyInterpreterTest(PallasBaseTest):
  # TODO(b/346651778): Support non-interpret mode checkify.
  INTERPRET = True

  def test_no_checkify(self,):
    def kernel(y_ref):
      y_ref[...] = jnp.zeros_like(y_ref[...])
    out_shape = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    pallas_call = self.pallas_call(kernel,
                                   out_shape=out_shape)
    checked_call = checkify.checkify(pallas_call)
    err, result = checked_call()
    err.throw()  # Should not raise.
    np.testing.assert_allclose(result, jnp.zeros_like(result))

  def test_does_not_clobber_previous_error(self,):
    def kernel(y_ref):
      y_ref[...] = jnp.zeros_like(y_ref[...])
      checkify.check(False, "error in kernel")
    out_shape = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    pallas_call = self.pallas_call(kernel,
                                   out_shape=out_shape)
    def error_before_call():
      checkify.check(False, "error before call")
      return pallas_call()
    checked_call = checkify.checkify(error_before_call)
    err, result = checked_call()
    with self.assertRaisesRegex(
          checkify.JaxRuntimeError, "error before call"):
      err.throw()
    np.testing.assert_allclose(result, jnp.zeros_like(result))

  @parameterized.parameters((False,), (True,))
  def test_trivial_check(self, assert_cond):
    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...]
      checkify.check(assert_cond, "pallas check failed")
    input = jnp.arange(4, dtype=jnp.int32)
    out_shape = jax.ShapeDtypeStruct(input.shape, input.dtype)
    pallas_call = self.pallas_call(kernel,
                                   out_shape=out_shape)
    checked_call = checkify.checkify(pallas_call)
    err, result = checked_call(input)
    if not assert_cond:
      with self.assertRaisesRegex(
            checkify.JaxRuntimeError, "pallas check failed"):
        err.throw()
    np.testing.assert_allclose(result, input)

  def test_nan_error(self):
    def kernel(x_ref, y_ref):
      y_ref[...] = jnp.log(x_ref[...])
    input = jnp.arange(4, dtype=jnp.float32) - 2
    out_shape = jax.ShapeDtypeStruct(input.shape, input.dtype)
    pallas_call = self.pallas_call(kernel,
                                   out_shape=out_shape)
    checked_call = checkify.checkify(pallas_call,
                                       errors=checkify.all_checks)
    err, result = checked_call(input)
    with self.assertRaisesRegex(
          checkify.JaxRuntimeError, "nan generated by primitive: log"):
      err.throw()
    is_nan = jnp.isnan(result)
    np.testing.assert_allclose(is_nan, input < 0)

  def test_nan_error_with_assertion(self):
    # TODO(b/346842088): Fix check asserts clobbering other errors.
    self.skipTest('Known failure.')
    # Test NaN error is not clobbered by an assertion failure
    def kernel(x_ref, y_ref):
      y_ref[...] = jnp.log(x_ref[...])
      checkify.check(False, "do not raise")
    input = jnp.arange(4, dtype=jnp.float32) - 10
    out_shape = jax.ShapeDtypeStruct(input.shape, input.dtype)
    pallas_call = self.pallas_call(kernel,
                                     out_shape=out_shape)
    checked_call = checkify.checkify(pallas_call,
                                       errors=checkify.all_checks)
    err, _ = checked_call(input)
    with self.assertRaisesRegex(
          checkify.JaxRuntimeError, "nan generated by primitive: log"):
      err.throw()

  @parameterized.parameters((5, 0), (8, 3), (4, 3))
  def test_checkify_returns_first_error_in_grid(
      self, num_loops, fail_iteration):
    # Check that checkify returns the first error that occurs
    # TODO(justinfu): This test doesn't make sense on GPU, where threads run
    # in parallel. Update checkify to return a grid of errors.
    def kernel(x_ref, _):
      value = jnp.squeeze(x_ref[...])
      checkify.check(
          value < fail_iteration, "failed on loop {itr}", itr=value)
    input_arr = jnp.arange(num_loops, dtype=jnp.float32)
    in_specs = [pl.BlockSpec((1,), lambda x: (x,))]
    out_specs = pl.BlockSpec((1,), lambda x: (x,))
    out_shape = jax.ShapeDtypeStruct((1,), dtype=jnp.float32)
    pallas_call = self.pallas_call(kernel,
                                 grid=(num_loops,),
                                 in_specs=in_specs,
                                 out_specs=out_specs,
                                 out_shape=out_shape)

    checked_call = checkify.checkify(pallas_call,
                                     errors=checkify.all_checks)
    err, _ = checked_call(input_arr)
    with self.assertRaisesRegex(
        checkify.JaxRuntimeError, f"failed on loop {fail_iteration}"):
      err.throw()


class PallasCallNamedGridTest(PallasBaseTest):
  def test_named_grid(self):

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...]

    x = jnp.arange(2 * 8 * 128, dtype=np.int32).reshape((2, 8, 128))
    y = self.pallas_call(
        kernel,
        out_shape=x,
        in_specs=[
            pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0)),
        grid=(("i", 2),)
    )(x)
    np.testing.assert_array_equal(y, x)

  def test_named_grid_reordered_names(self):

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...]

    x = jnp.arange(4 * 16 * 128, dtype=np.int32).reshape((4, 16, 128))
    y = self.pallas_call(
        kernel,
        out_shape=x,
        in_specs=[
            pl.BlockSpec((None, 8, 128), lambda i, j: (i, j, 0)),
        ],
        out_specs=pl.BlockSpec((None, 8, 128), lambda i, j: (i, j, 0)),
        grid=(("j", 4), ("i", 2))
    )(x)
    np.testing.assert_array_equal(y, x)

  def test_can_query_named_grid_size_in_kernel_via_psum(self):

    def kernel(x_ref, y_ref):
      self.assertEqual(lax.psum(1, "i"), 2)
      self.assertEqual(lax.psum(1, "j"), 4)
      y_ref[...] = x_ref[...]

    x = jnp.arange(4 * 16 * 128, dtype=np.int32).reshape((4, 16, 128))
    y = self.pallas_call(
        kernel,
        out_shape=x,
        in_specs=[
            pl.BlockSpec((None, 8, 128), lambda i, j: (i, j, 0)),
        ],
        out_specs=pl.BlockSpec((None, 8, 128), lambda i, j: (i, j, 0)),
        grid=(("j", 4), ("i", 2))
    )(x)
    np.testing.assert_array_equal(y, x)

  def test_can_query_named_dynamic_grid_size_in_kernel_via_psum(self):
    # TODO(): Enable dynamic grid size via axis_size primitive.
    self.skipTest("Not supported.")

    def kernel(x_ref, y_ref):
      self.assertEqual(lax.psum(1, "i"), 2)
      self.assertEqual(lax.psum(1, "j"), 4)
      y_ref[...] = x_ref[...]

    x = jnp.arange(4 * 8 * 128, dtype=np.int32).reshape((4, 8, 128))
    @jax.jit
    def foo(n):
      return self.pallas_call(
          kernel,
          out_shape=x,
          in_specs=[
              pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0)),
          ],
          out_specs=pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0)),
          grid=(("i", n),)
      )(x)
    y = foo(4)
    np.testing.assert_array_equal(y, x)

  def test_can_query_named_grid_program_id_in_kernel_via_axis_index(self):
    if self.INTERPRET:
      self.skipTest("Not supported in interpret mode.")
    def kernel(x_ref, y_ref):
      i_index = lax.axis_index("i")
      y_ref[...] = x_ref[...] + i_index

    x = jnp.arange(4 * 8 * 128, dtype=np.int32).reshape((4, 8, 128))
    y = self.pallas_call(
        kernel,
        out_shape=x,
        in_specs=[
            pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0)),
        grid=(("i", 4),),
    )(x)
    np.testing.assert_array_equal(
        y, x + jnp.arange(4, dtype=jnp.int32)[:, None, None]
    )


class PallasCallNamedGridInterpretTest(PallasCallNamedGridTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main()
