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

import contextlib
import functools
import itertools
import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax import random
from jax._src import checkify
from jax._src import config
from jax._src import linear_util as lu
from jax._src import state
from jax._src import test_util as jtu
from jax._src.lax.control_flow.for_loop import for_loop
from jax._src.pallas.pallas_call import _trace_to_jaxpr
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.gpu import attention
from jax.experimental.pallas.ops.gpu import layer_norm
from jax.experimental.pallas.ops.gpu import rms_norm
from jax.experimental.pallas.ops.gpu import softmax
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas import gpu as plgpu
else:
  plgpu = None


# TODO(sharadmv): Update signatures of pallas_call to correct inputs/outputs.
# pylint: disable=no-value-for-parameter

config.parse_flags_with_absl()

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
    pid = pl.program_id(axis=0)
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
      pl.pallas_call, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      in_specs=[
        pl.BlockSpec(lambda i, _: (i, 0), (bm, x.shape[1])),
        pl.BlockSpec(lambda _, j: (0, j), (y.shape[0], bn))
      ],
      out_specs=pl.BlockSpec(lambda i, j: (i, j), (bm, bn)),
      grid=(pl.cdiv(m, bm), pl.cdiv(n, bn)))
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
class PallasTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")
    if not self.INTERPRET:
      if not jtu.test_device_matches(["gpu"]):
        self.skipTest("Only works on GPU")
      if (jtu.test_device_matches(["cuda"]) and
          not jtu.is_cuda_compute_capability_at_least("8.0")):
        self.skipTest("Only works on GPU with capability >= sm80")
      if sys.platform == "win32":
        self.skipTest("Only works on non-Windows platforms")

    super().setUp()
    _trace_to_jaxpr.cache_clear()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


class PallasCallTest(PallasTest):

  def test_add_one(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32))
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1.

    x = 0.
    self.assertEqual(add_one(x), 1.)

  def test_add_singleton_vector(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
        grid=1)
    def add_one(x_ref, o_ref):
      o_ref[0] = x_ref[0] + 1.

    x = jnp.array([0.], jnp.float32)
    np.testing.assert_allclose(add_one(x), jnp.array([1.], jnp.float32))

  def test_add_vector_block_spec(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), jnp.int32),
        in_specs=[pl.BlockSpec(lambda i: i, (1,))],
        out_specs=pl.BlockSpec(lambda i: i, (1,)),
        grid=8, debug=False)
    def add_one(x_ref, o_ref):
      o_ref[0] = x_ref[0] + 1

    np.testing.assert_allclose(add_one(jnp.arange(8)), jnp.arange(8) + 1)

  def test_add_matrix_block_spec(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.int32),
        in_specs=[pl.BlockSpec(lambda i, j: (i, j), (2, 2))],
        out_specs=pl.BlockSpec(lambda i, j: (i, j), (2, 2)),
        grid=(4, 4))
    def add_one(x_ref, o_ref):
      o_ref[:, :] = x_ref[:, :] + 1

    x = jnp.arange(64).reshape((8, 8))
    np.testing.assert_allclose(add_one(x), x + 1)

  def test_bool_array(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.bool_))
    def logical_and(x_ref, o_ref):
      o_ref[()] = jnp.logical_and(x_ref[()], True)

    x = jnp.array(True)
    self.assertTrue(jnp.all(logical_and(x)))

  def test_vector_indexing(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def index(x_ref, i_ref, o_ref):
      o_ref[()] = x_ref[i_ref[()]]

    x = jnp.arange(5.)
    for i in range(5):
      np.testing.assert_allclose(index(x, i), x[i])

  def test_hoisted_consts(self):
    # See https://github.com/google/jax/issues/21557.
    x = jnp.zeros(32)
    indices = jnp.arange(4).reshape((2, 2))

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    def kernel(src, dst):
      dst[indices] = src[indices]

    jax.block_until_ready(kernel(x))

  def test_vector_slicing(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
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

  def test_pallas_call_with_input_output_aliasing(self):

    def add_inplace_kernel(_, o_ref, *, block_size):
      pid = pl.program_id(axis=0)  # we use a 1d launch grid so axis is 0
      block_start = pid * block_size
      offsets = block_start + jnp.arange(block_size)
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
    m, n = 32, 4
    out_shape = jax.ShapeDtypeStruct((4, n), jnp.float32)
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


class PallasCallInterpreterTest(PallasCallTest):
  INTERPRET = True


class PallasControlFlowTest(PallasTest):

  def setUp(self):
    if self.INTERPRET:
      self.skipTest("Control flow not supported in interpreter mode yet.")

    super().setUp()

  def test_loop_with_float64_carry(self):
    # Test that the jnp.zeros(f64) loop init_val is actually f64, and that
    # fori_loop handles i64 index variables, i.e. error: 'scf.for' op  along
    # control flow edge from Region #0 to Region #0: source type #0
    # 'tensor<4xf64>' should match input type #0 'tensor<4xf32>'
    with config.enable_x64(True):
      @functools.partial(self.pallas_call,
                         out_shape=jax.ShapeDtypeStruct((4,), jnp.float64),
                         grid=1,
                         debug=False)
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
    arg = jnp.float32(0.)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       debug=False)
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
    arg = jnp.float32(0.)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       grid=1,
                       debug=False)
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
    arg = jnp.float32([0.] * 8)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       in_specs=[pl.BlockSpec(lambda _: (), ()),
                                 pl.BlockSpec(lambda i: i, (block_size,))],
                       out_specs=pl.BlockSpec(lambda i: i, (block_size,)),
                       grid=pl.cdiv(arg.shape[0], block_size),
                       debug=False)
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
    arg = jnp.float32([0.] * 8)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       in_specs=[pl.BlockSpec(lambda _: (), ()),
                                 pl.BlockSpec(lambda i: i, (block_size,))],
                       out_specs=pl.BlockSpec(lambda i: i, (block_size,)),
                       grid=pl.cdiv(arg.shape[0], block_size),
                       debug=False)
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
    x = jnp.arange(64.)[:, None]
    y = jnp.arange(128.)[None, :]
    # TODO(sharadmv): Renaming in_specs->in_spec silently breaks.
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), jnp.float32),
        in_specs=[
            pl.BlockSpec(lambda _, __: (), ()),
            pl.BlockSpec(lambda i, _: (i, 0), (bx, 1)),
            pl.BlockSpec(lambda _, j: (0, j), (1, by))],
        out_specs=pl.BlockSpec(lambda i, j: (i, j), (bx, by)),
        grid=(pl.cdiv(x.shape[0], bx), pl.cdiv(y.shape[1], by)),
        debug=False)
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

  def test_conditional_write(self):
    arg = jnp.arange(8, dtype=jnp.float32)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       debug=False)
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
    program = jnp.int32([0, 1, 2, 3, 2])
    params = jnp.arange(len(program) * 3.).reshape(len(program), 3)
    x = jnp.arange(7.)
    bx = 4

    @jax.jit
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
        in_specs=[
            pl.BlockSpec(lambda _: (0,), program.shape),  # program
            pl.BlockSpec(lambda _: (0, 0), params.shape),  # params
            pl.BlockSpec(lambda i: (i,), (bx,))],  # x
        out_specs=pl.BlockSpec(lambda i: (i,), (bx,)),
        grid=pl.cdiv(x.shape[0], bx),
        debug=False)
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
          (jnp.zeros(x.shape), program_ref, params_ref))[0]

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
    # ** Difference is the closure over params_ref in the switch branches. **
    program = jnp.int32([0, 1, 2, 3, 2, -1])
    params = jnp.arange(len(program) * 3.).reshape(len(program), 3)
    x = jnp.arange(7.)
    bx = 4

    @jax.jit
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
        in_specs=[
            pl.BlockSpec(lambda _: (0,), program.shape),  # program
            pl.BlockSpec(lambda _: (0, 0), params.shape),  # params
            pl.BlockSpec(lambda i: (i,), (bx,))],  # x
        out_specs=pl.BlockSpec(lambda i: (i,), (bx,)),
        grid=pl.cdiv(x.shape[0], bx),
        debug=False)
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
          (jnp.zeros(x.shape), program_ref, params_ref))[0]

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

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((8,), jnp.int32))
    def f(y_ref):
      def body(i, _):
        y_ref[i] = i
      lax.fori_loop(0, y_ref.shape[0], body, None)
    y = f()
    np.testing.assert_allclose(y, jnp.arange(8))

  def test_fori_loop_with_dynamic_indices(self):

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

class PallasCallAutodifferentiationTest(PallasTest):

  @parameterized.named_parameters(*AD_TEST_CASES)
  def test_jvp(self, impl):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        debug=False,
        grid=1)
    def pallas_impl(x_ref, o_ref):
      x = x_ref[()]
      o_ref[()] = impl(x)

    k1, k2 = random.split(random.key(0))
    x = random.normal(k1)
    t = random.normal(k2)
    out_primal, out_tangent = jax.jvp(pallas_impl, (x,), (t,))
    out_primal_ref, out_tangent_ref = jax.jvp(impl, (x,), (t,))
    np.testing.assert_allclose(out_primal, out_primal_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(out_tangent, out_tangent_ref, atol=1e-5,
                               rtol=1e-5)
    jtu.check_grads(pallas_impl, (x,), modes=["fwd"], order=2)

  @parameterized.named_parameters(*AD_TEST_CASES)
  def test_pallas_around_grad(self, impl):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        name=self.id().split(".")[-1],
        debug=False,
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
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        debug=False,
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
    np.testing.assert_allclose(out_primal, out_primal_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(out_tangent, out_tangent_ref, atol=1e-5,
                               rtol=1e-5)
    jtu.check_grads(pallas_impl, (x,), modes=["fwd"], order=2)

  # TODO(sharadmv): enable this when we update Triton
  # def test_jvp_matmul(self):
  #   k1, k2 = random.split(random.key(0))
  #   x = random.normal(k1, (256, 128))
  #   y = random.normal(k2, (128, 64))
  #   bm, bn, bk, gm = 64, 128, 32, 8
  #   mm = functools.partial(matmul, bm=bm, bn=bn, bk=bk, gm=gm,
  #                          interpret=self.INTERPRET)
  #   jtu.check_grads(mm, (x, y), modes=["fwd"], order=1)

  def test_slicing_block_spec(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        in_specs=[
          pl.BlockSpec(lambda _: (0, 0), (None, 4)),
          pl.BlockSpec(lambda _: (1, 0), (None, 4)),
        ],
        debug=False, grid=1)
    def add_vectors(x_ref, y_ref, o_ref):
      o_ref[:] = x_ref[:] + y_ref[:]
    xy = jnp.arange(8.).reshape((2, 4))
    out = add_vectors(xy, xy)
    out_ref = xy[0] + xy[1]
    np.testing.assert_allclose(out, out_ref)


class PallasCallVmapTest(PallasTest):

  def test_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(add_one)(jnp.arange(8))
    out_ref = jnp.arange(1, 9)
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_simple_kernel_with_in_axes_None(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False)
    def add(x_ref, y_ref, o_ref):
      o_ref[()] = x_ref[()] + y_ref[()]
    out = jax.vmap(add, in_axes=(0, None))(jnp.arange(8), 1)
    out_ref = jnp.arange(1, 9)
    np.testing.assert_allclose(out, out_ref)

  def test_double_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(jax.vmap(add_one))(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_quadruple_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(jax.vmap(jax.vmap(jax.vmap(add_one))))(
        jnp.arange(15 * 8).reshape((5, 3, 4, 2)))
    out_ref = jnp.arange(1, 15 * 8 + 1).reshape((5, 3, 4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_quadruple_vmap_of_batched_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((7,), jnp.int32),
        debug=False,
        grid=(7,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    out = jax.vmap(jax.vmap(jax.vmap(jax.vmap(add_one))))(
        jnp.arange(15 * 8 * 7).reshape((5, 3, 4, 2, 7)))
    out_ref = jnp.arange(1, 15 * 8 * 7 + 1).reshape((5, 3, 4, 2, 7))
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_slicing_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        debug=False,
        grid=(2,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    out = jax.vmap(add_one)(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_kernel_with_input_output_aliases(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False,
        input_output_aliases={1:0},
        grid=())
    def add(x_ref, _, o_ref):
      o_ref[()] = x_ref[()] + o_ref[()] + 1
    out = jax.vmap(add, in_axes=(0, None))(jnp.arange(8), 1)
    out_ref = jnp.arange(2, 10)
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_kernel_with_input_output_aliases_different_axes(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.int32),
        debug=False,
        input_output_aliases={0: 0},
        grid=(),
    )
    def add(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1

    out = jax.vmap(add, in_axes=1)(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2)).swapaxes(0, 1)
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_slicing_kernel_different_axes(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        debug=False,
        grid=(2,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    add_one_ref = lambda x: x + 1
    x = jnp.arange(8).reshape((2, 4))

    out = jax.vmap(add_one, in_axes=1, out_axes=1)(x)
    out_ref = jax.vmap(add_one_ref, in_axes=1, out_axes=1)(x)
    np.testing.assert_allclose(out, out_ref)

    out = jax.vmap(add_one, in_axes=1, out_axes=0)(x)
    out_ref = jax.vmap(add_one_ref, in_axes=1, out_axes=0)(x)
    np.testing.assert_allclose(out, out_ref)

  def test_double_vmap_of_slicing_kernel_different_axes(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        debug=False,
        grid=(4,))
    def sin(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = jnp.sin(x_ref[i])
    sin_ref = jnp.sin
    x = jnp.arange(64.).reshape((8, 4, 2))

    out = jax.vmap(jax.vmap(sin, in_axes=1), in_axes=0)(x)
    out_ref = jax.vmap(jax.vmap(sin_ref, in_axes=1), in_axes=0)(x)
    np.testing.assert_allclose(out, out_ref, atol=1e-3, rtol=1e-3)

  def test_small_large_vmap(self):
    # Catches https://github.com/google/jax/issues/18361
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        debug=False,
        grid=(2,))
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1

    add_one = jax.vmap(jax.vmap(add_one))
    add_one_ref = lambda x: x + 1

    x = random.randint(random.key(0), (4, 65536, 2), 0, 10000)

    out = add_one(x)
    out_ref = add_one_ref(x)

    np.testing.assert_allclose(out, out_ref)

  def test_small_small_large_vmap(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        debug=False,
        grid=(2,))
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1

    add_one = jax.vmap(jax.vmap(jax.vmap(add_one)))
    add_one_ref = lambda x: x + 1

    x = random.randint(random.key(0), (2, 2, 65536, 2), 0, 10000)

    out = add_one(x)
    out_ref = add_one_ref(x)

    np.testing.assert_allclose(out, out_ref)


class PallasCallInterpreterVmapTest(PallasCallVmapTest):
  INTERPRET = True


class PallasOpsTest(PallasTest):

  ELEMENTWISE_OPS = [
      (
          [jnp.abs, jnp.negative],
          ["int16", "int32", "int64", "float16", "float32", "float64"],
      ),
      ([jnp.ceil, jnp.floor], ["float32", "float64"]),
      (
          [jnp.exp, jnp.exp2, jnp.sin, jnp.cos, jnp.log, jnp.sqrt],
          ["float16", "float32", "float64"],
      ),
      (
          # fmt: off
          [jnp.expm1, jnp.log1p, jnp.cbrt, lax.rsqrt, jnp.tan, jnp.asin,
           jnp.acos, jnp.atan, jnp.sinh, jnp.cosh, jnp.asinh, jnp.acosh,
           jnp.atanh],
          # fmt: on
          ["float32", "float64"],
      ),
      ([lax.population_count, lax.clz, jnp.invert], ["int32", "int64"]),
  ]

  @parameterized.named_parameters(
      (f"{fn.__name__}_{dtype}", fn, dtype)
      for args in ELEMENTWISE_OPS
      for fn, dtype in itertools.product(*args)
  )
  def test_elementwise(self, fn, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), dtype), grid=1
    )
    def kernel(x_ref, o_ref):
      o_ref[:] = fn(x_ref[...])

    with contextlib.ExitStack() as stack:
      if jnp.dtype(dtype).itemsize == 8:
        stack.enter_context(config.enable_x64(True))
      x = jnp.array([0.42, 2.4]).astype(dtype)
      np.testing.assert_allclose(kernel(x), fn(x), rtol=1e-6)

  @parameterized.parameters(
      ("float32", "int32"),
      ("float64", "int32"),
      ("float32", "float32"),
      ("float64", "float64"),
  )
  def test_pow(self, x_dtype, y_dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), x_dtype), grid=1
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[:] = lax.pow(x_ref[...], y_ref[...])

    with contextlib.ExitStack() as stack:
      if jnp.dtype(x_dtype).itemsize == 8:
        stack.enter_context(config.enable_x64(True))
      x = jnp.array([1, 2, 3, 4]).astype(x_dtype)
      y = jnp.array([1, 2, 3, 4]).astype(y_dtype)
      np.testing.assert_allclose(kernel(x, y), lax.pow(x, y))

  @parameterized.parameters(0, 1, 2, 3, 4, 5, -1, -2, -3)
  def test_integer_pow(self, y):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[:] = lax.integer_pow(x_ref[...], y)

    x = jnp.array([1, 2, 3, 4]).astype(jnp.float32) / 10
    np.testing.assert_allclose(kernel(x), lax.integer_pow(x, y))

  @parameterized.parameters("float32", "float64")
  def test_nextafter(self, dtype):
    if jtu.test_device_matches(["tpu"]) and dtype == "float64":
      self.skipTest("float64 disabled on TPU.")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), dtype), grid=1
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[:] = jnp.nextafter(x_ref[...], y_ref[...])

    with contextlib.ExitStack() as stack:
      if jnp.dtype(dtype).itemsize == 8:
        stack.enter_context(config.enable_x64(True))
      x = jnp.array([1, 2, 3, 4]).astype(dtype)
      y = jnp.array([1, 2, 3, 4]).astype(dtype)
      np.testing.assert_allclose(kernel(x, y), jnp.nextafter(x, y))

  COMPARISON_OPS = [
      jnp.equal,
      jnp.not_equal,
      jnp.less,
      jnp.less_equal,
      jnp.greater,
      jnp.greater_equal,
  ]

  @parameterized.named_parameters(
      (f"{fn.__name__}_{dtype}", fn, dtype)
      for fn, dtype in itertools.product(
          COMPARISON_OPS, ["int32", "uint32", "float16", "float32"]
      )
  )
  def test_comparison(self, fn, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), jnp.bool_),
        grid=1)
    def kernel(x_ref, y_ref, o_ref):
      o_ref[:] = fn(x_ref[...], y_ref[...])

    x = jnp.array([1, 3, -4, -6, 2, 5, 4, -7]).astype(dtype)
    y = jnp.array([3, 1, -4, -5, 2, -2, 2, 4]).astype(dtype)
    np.testing.assert_allclose(kernel(x, y), fn(x, y))

  def test_isnan(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), jnp.bool_),
        grid=1)
    def isnan(x_ref, o_ref):
      o_ref[:] = jnp.isnan(x_ref[...])

    x = jnp.arange(8.)
    x = x.at[3].set(jnp.nan)
    np.testing.assert_allclose(isnan(x), jnp.isnan(x))

  @parameterized.parameters(
      ("int32", "float32"),
      ("float32", "float32"),
  )
  def test_true_divide(self, dtype, out_dtype):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8,), out_dtype),
        grid=1,
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = jnp.true_divide(x_ref[...], y_ref[...])

    x = jnp.array([1, 3, -4, -6, 2, 5, 4, -7]).astype(dtype)
    y = jnp.array([3, 1, -4, -5, 2, -2, 2, 4]).astype(dtype)
    np.testing.assert_allclose(jnp.true_divide(x, y), kernel(x, y))

  @parameterized.parameters("float16", "bfloat16")
  def test_true_divide_unsupported(self, dtype):
    if self.INTERPRET:
      self.skipTest("No lowering in interpreter mode")

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), dtype),
        grid=1,
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = jnp.true_divide(x_ref[...], y_ref[...])

    x = jnp.array([2.4, 4.2]).astype(dtype)
    y = jnp.array([4.2, 2.4]).astype(dtype)
    with self.assertRaises(Exception):
      kernel(x, y)

  BINARY_OPS = [
      ([jnp.floor_divide], ["int32", "uint32"]),
      (
          [jnp.add, jnp.subtract, jnp.multiply],
          ["int16", "int32", "uint32", "float16", "float32"],
      ),
      ([jnp.remainder], ["int32", "uint32", "float32"]),
      (
          # fmt: off
          [jnp.bitwise_and, jnp.bitwise_or, jnp.bitwise_xor,
           jnp.bitwise_left_shift, jnp.bitwise_right_shift],
          # fmt: on
          ["int32", "uint32"],
      ),
  ]

  @parameterized.named_parameters(
      (f"{fn.__name__}_{dtype}", fn, dtype)
      for args in BINARY_OPS
      for fn, dtype in itertools.product(*args)
  )
  def test_binary(self, f, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), dtype), grid=1
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = f(x_ref[...], y_ref[...])

    x = jnp.array([1, 3, -4, -6, 2, 5, 4, -7]).astype(dtype)
    if (f == jnp.bitwise_left_shift):
      y = jnp.array([3, 1, 4, 5, 2, 2, 2, 4]).astype(dtype)
    else:
      y = jnp.array([3, 1, -4, -5, 2, -2, 2, 4]).astype(dtype)

    np.testing.assert_allclose(f(x, y), kernel(x, y))

  @parameterized.parameters(
      ((8, 4), jnp.int32, 0),
      ((8, 16), jnp.float32, 1),
      ((8, 16, 2), jnp.int8, 1),
  )
  def test_broadcasted_iota(self, shape, dtype, dimension):
    f = lambda: jax.lax.broadcasted_iota(dtype, shape, dimension)

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct(shape, dtype), grid=1
    )
    def kernel(o_ref):
      o_ref[...] = f()

    np.testing.assert_allclose(f(), kernel())

  @parameterized.parameters("float16", "bfloat16", "float32")
  def test_approx_tanh(self, dtype):
    if self.INTERPRET:
      self.skipTest("approx_tanh is not supported in interpreter mode")
    if (dtype == "bfloat16" and
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("tanh.approx.bf16 requires a GPU with capability >= sm90")

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), dtype), grid=1
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plgpu.approx_tanh(x_ref[...])

    x = jnp.asarray([-1, 0.42, 0.24, 1]).astype(dtype)
    # We upcast to float32 because NumPy <2.0 does not handle custom dtypes
    # properly. See https://github.com/google/jax/issues/11014.
    np.testing.assert_allclose(
        kernel(x).astype(jnp.float32),
        jnp.tanh(x).astype(jnp.float32),
        atol=5e-3,
        rtol=5e-3,
    )

  def test_elementwise_inline_asm(self):
    if self.INTERPRET:
      self.skipTest(
          "elementwise_inline_asm is not supported in interpreter mode"
      )

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float16),
        grid=1,
    )
    def kernel(x_ref, o_ref):
      [o_ref[...]] = plgpu.elementwise_inline_asm(
          "tanh.approx.f16x2 $0, $1;",
          args=[x_ref[...]],
          constraints="=r,r",
          pack=2,
          result_shape_dtypes=[jax.ShapeDtypeStruct(x_ref.shape, x_ref.dtype)],
      )

    x = jnp.arange(256).astype(jnp.float16)
    np.testing.assert_allclose(kernel(x), jnp.tanh(x), atol=5e-3, rtol=5e-3)

  def test_debug_print(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        grid=1,
        compiler_params=dict(triton=dict(num_warps=1, num_stages=1))
    )
    def kernel(x_ref, o_ref):
      pl.debug_print("It works!")

    x = jnp.array([4.2, 2.4]).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
      jax.effects_barrier()

    self.assertIn("It works!", output())

  def test_debug_print_with_values(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        grid=1,
        compiler_params=dict(triton=dict(num_warps=1, num_stages=1))
    )
    def kernel(x_ref, o_ref):
      pl.debug_print("x[0] =", x_ref[0])

    x = jnp.array([4.2, 2.4]).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
      jax.effects_barrier()

    self.assertIn("x[0] = 4.2", output())

  @parameterized.parameters(
      ((2, 4), (8,)),
      ((2, 4), (8, 1)),
      ((2, 4), (1, 8)),
      ((64,), (32, 2)),
  )
  def test_reshape(self, in_shape, out_shape):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1,
    )
    def f(x_ref, o_ref):
      o_ref[...] = x_ref[...].reshape(out_shape)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = x.reshape(out_shape)
    np.testing.assert_allclose(f(x), expected)

  @parameterized.parameters(
      # fmt: off
      ((), (1,)),
      ((), (1, 1)),
      ((2, 4), (2, 4)),
      ((2, 4), (2, 4, 1)),
      ((2, 4, 1), (2, 4)),
      ((2, 4), (1, 2, 4)),
      ((1, 2, 4), (2, 4)),
      ((2, 4), (2, 1, 4)),
      ((1, 2, 1, 4, 1), (2, 4)),
      ((2, 4,), (1, 2, 1, 4)),
      ((2, 4,), (1, 2, 4, 1)),
      ((1, 2, 4, 1), (1, 2, 1, 4, 1)),
      # fmt: on
  )
  def test_reshape_noop_or_singleton_dims(self, in_shape, out_shape):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1,
    )
    def f(x_ref, o_ref):
      o_ref[...] = x_ref[...].reshape(out_shape)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = x.reshape(out_shape)
    np.testing.assert_allclose(f(x), expected)

  def test_num_programs(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.int32),
        grid=4,
    )
    def kernel(o_ref):
      o_ref[pl.program_id(0)] = pl.num_programs(0)

    np.testing.assert_array_equal(
        kernel(), np.asarray([4, 4, 4, 4], dtype=np.int32)
    )

  def test_where_broadcasting(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4, 2, 2), jnp.float32),
        grid=1,
    )
    def copyitem(x_ref, in_idx_ref, out_idx_ref, o_ref):
      mask = (jnp.arange(o_ref.shape[0]) == out_idx_ref[()])[:, None, None]
      o_ref[...] = jnp.where(mask, x_ref[in_idx_ref[()]], 0)

    x = jnp.arange(7 * 2 * 2.0).reshape(7, 2, 2)
    for ii in range(7):
      for oi in range(4):
        out = copyitem(x, ii, oi)
        self.assertEqual((4, 2, 2), out.shape)
        np.testing.assert_allclose(out[:oi], jnp.zeros_like(out[:oi]))
        np.testing.assert_allclose(out[oi], x[ii])
        np.testing.assert_allclose(out[oi + 1 :], jnp.zeros_like(out[oi + 1 :]))

  @parameterized.parameters(
      ((), (2,), ()),
      ((1,), (2,), (0,)),
      ((1, 1), (2, 2), (0, 1)),
      ((), (2, 2), ()),
  )
  def test_broadcast_in_dim(self, in_shape, out_shape, dims):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1,
    )
    def f(x_ref, o_ref):
      x = x_ref[...]
      o_ref[...] = jax.lax.broadcast_in_dim(x, out_shape, dims)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = jax.lax.broadcast_in_dim(x, out_shape, dims)
    np.testing.assert_allclose(f(x), expected)

  @parameterized.product(
      size=[16, 32, 64],
      dtype=["float32", "float16"],
      trans_x=[False, True],
      trans_y=[False, True],
  )
  def test_dot(self, size, dtype, trans_x, trans_y):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((size, size), dtype),
        grid=1,
    )
    def dot(x_ref, y_ref, o_ref):
      x = x_ref[:, :]
      y = y_ref[:, :]
      o_ref[:, :] = pl.dot(x, y, trans_x, trans_y).astype(o_ref.dtype)

    k1, k2 = random.split(random.key(0))
    x = random.normal(k1, (size, size), dtype=dtype)
    y = random.normal(k2, (size, size), dtype=dtype)
    out = dot(x, y)
    expected = jnp.dot(x.T if trans_x else x, y.T if trans_y else y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.product(
      size=[1, 2, 64, 129, 1021],
      block_size=[1, 2, 32, 64, 128],
  )
  def test_masked_load_store(self, size, block_size):
    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((size,), jnp.float32)),
        grid=pl.cdiv(size, block_size),
    )
    def kernel(x_ref, o_ref):
      idx = pl.program_id(0) * block_size + jnp.arange(block_size)
      mask = idx < x_ref.shape[0]
      x = pl.load(x_ref, (idx,), mask=mask)
      pl.store(o_ref, (idx,), x + 1.0, mask=mask)

    key = random.key(0)
    x = random.normal(key, (size,))
    np.testing.assert_allclose(kernel(x), x + 1.0, atol=1e-5, rtol=1e-5)

  def test_masked_oob_load_store_slice(self):
    n = 16

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((n,), jnp.float32)),
        grid=1,
    )
    def masked_oob_load_store_slice(x_ref, mask_ref, start_idx_ref, o_ref):
      x = pl.load(x_ref, (pl.dslice(start_idx_ref[()], n)),
                  mask=mask_ref[:], other=-1.)
      pl.store(o_ref, (pl.dslice(None),), x)

    x = random.normal(random.key(0), (n,))
    slice_start = random.randint(random.key(2), (), 1, n)
    indices = jnp.arange(n) + slice_start
    mask = indices < n
    out = masked_oob_load_store_slice(x, mask, slice_start)
    o_new = jnp.where(mask, x[indices], jnp.full_like(x, -1.))
    np.testing.assert_array_equal(out, o_new)

  def test_strided_load(self):
    if self.INTERPRET:
      # TODO(b/329733289): Remove this once the bug is fixed.
      self.skipTest("Strided load not yet supported in interpreter mode")

    # Reproducer from https://github.com/google/jax/issues/20895.
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[::4]

    x = jnp.arange(16, dtype=jnp.float32)
    np.testing.assert_array_equal(kernel(x), x[::4])

  def test_broadcasted_load_store(self):
    m, n = 16, 32

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((m, n), jnp.float32)),
        grid=1,
    )
    def load(x_ref, o_ref):
      x = pl.load(x_ref, (jnp.arange(m)[:, None], jnp.arange(n)[None, :]))
      pl.store(o_ref, (jnp.arange(m)[:, None], jnp.arange(n)[None, :]), x + 1.0)

    key = random.key(0)
    x = random.normal(key, (m, n))
    np.testing.assert_allclose(load(x), x + 1.0, atol=1e-5, rtol=1e-5)

  @parameterized.parameters(
      ((16, 32), (16,)),
      ((16, 32), (32,)),
      ((16, 32), (16, 31)),
  )
  def test_invalid_broadcasted_load(self, x_shape, mask_shape):
    if self.INTERPRET:
      self.skipTest("No broadcasting checks in pl.load in interpreter mode")

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32)
    )
    def kernel(x_ref, mask_ref, o_ref):
      del o_ref  # Unused.
      pl.load(x_ref, slice(None), mask=mask_ref[:])

    x = jnp.ones(x_shape, dtype=jnp.float32)
    mask = jnp.ones(mask_shape, dtype=jnp.bool_)
    # assertRaises* methods do not support inspecting the __cause__, so
    # we have to check it manually.
    try:
      kernel(x, mask)
    except Exception as e:
      self.assertIn("Cannot broadcast", str(e.__cause__))
    else:
      self.fail("Expected exception due to invalid broadcasting")

  def test_swap(self):
    m, n = 16, 32

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((m, n), jnp.float32),) * 2,
        grid=1,
        input_output_aliases={0: 0, 1: 1},
    )
    def swap(_, _2, x_ref, y_ref):
      x = x_ref[:]
      y = pl.swap(y_ref, (slice(None),), x)
      x_ref[:] = y

    x = random.normal(random.key(0), (m, n))
    y = random.normal(random.key(1), (m, n))
    out = swap(x, y)
    np.testing.assert_array_equal(out[0], y)
    np.testing.assert_array_equal(out[1], x)

  def test_masked_swap(self):
    m, n = 16, 32

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((m, n), jnp.float32),) * 2,
        grid=1,
        input_output_aliases={0: 0, 1: 1},
    )
    def masked_swap(_, _2, mask_ref, x_ref, y_ref):
      x = x_ref[:]
      y = pl.swap(y_ref, (slice(None),), x, mask=mask_ref[:])
      x_ref[:] = y

    x = random.normal(random.key(0), (m, n))
    y = random.normal(random.key(1), (m, n))
    mask = random.bernoulli(random.key(2), shape=(m, n))
    out = masked_swap(x, y, mask)
    np.testing.assert_array_equal(out[0], jnp.where(mask, y, x))
    np.testing.assert_array_equal(out[1], jnp.where(mask, x, y))

  def test_masked_oob_swap_slice(self):
    m, n = 32, 16

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((n,), jnp.float32),
                   jax.ShapeDtypeStruct((m,), jnp.float32)),
        grid=1,
        input_output_aliases={0: 0, 1: 1},
    )
    def masked_oob_swap_slice(_, _2, mask_ref, start_idx_ref, x_ref, y_ref):
      x, mask = x_ref[:], mask_ref[:]
      y = pl.swap(y_ref, (pl.dslice(start_idx_ref[()], n)), x, mask=mask)
      x_ref[:] = y

    x = random.normal(random.key(0), (n,))
    y = random.normal(random.key(1), (m,))
    slice_start = random.randint(random.key(2), (), m-n+1, m)
    indices = jnp.arange(n) + slice_start
    mask = indices < m
    out = masked_oob_swap_slice(x, y, mask, slice_start)

    # the unjittable masked indexing equivalent
    unmasked_idx = indices[mask]
    x_new = x.at[mask].set(y[unmasked_idx])
    y_new = y.at[unmasked_idx].set(x[mask])
    np.testing.assert_array_equal(out[0], x_new)
    np.testing.assert_array_equal(out[1], y_new)

  @parameterized.named_parameters(
      ("add_i32", pl.atomic_add, np.array([1, 2, 3, 4], np.int32), np.sum),
      ("max_i", pl.atomic_max, np.array([1, 2, 3, 4], np.int32), np.max),
      ("min_i32", pl.atomic_min, np.array([1, 2, 3, 4], np.int32), np.min),
      ("add_f16", pl.atomic_add, np.array([1, 2, 3, 4], np.float16), np.sum),
      ("add_f32", pl.atomic_add, np.array([1, 2, 3, 4], np.float32), np.sum),
      ("max_f32", pl.atomic_max, np.array([1, 2, 3, 4], np.float32), np.max),
      ("min_f32", pl.atomic_min, np.array([1, 2, 3, 4], np.float32), np.min),
  )
  def test_scalar_atomic(self, op, value, numpy_op):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), value.dtype),
        grid=value.shape[0],
        input_output_aliases={1: 0},
    )
    def atomic_kernel(x_ref, _, o_ref):
      pid = pl.program_id(axis=0)
      op(o_ref, (), x_ref[pid])

    if op == pl.atomic_add:
      neutral = np.array(0, dtype=value.dtype)
    elif op == pl.atomic_max:
      if np.issubdtype(value.dtype, np.integer):
        neutral = np.array(np.iinfo(value.dtype).min, value.dtype)
      else:
        neutral = np.array(-float("inf"), value.dtype)
    elif op == pl.atomic_min:
      if np.issubdtype(value.dtype, np.integer):
        neutral = np.array(np.iinfo(value.dtype).max, value.dtype)
      else:
        neutral = np.array(float("inf"), value.dtype)
    elif op == pl.atomic_or:
      neutral = np.array(False, value.dtype)
    else:
      raise NotImplementedError()
    out = atomic_kernel(value, neutral)
    np.testing.assert_allclose(out, numpy_op(value))

  @parameterized.parameters((0,), (1,))
  def test_array_atomic_add(self, axis):
    m, n = 32, 8
    if axis == 0:
      grid = m
    else:
      grid = n
    out_shape = jax.ShapeDtypeStruct((n if axis == 0 else m,), jnp.float32)

    @functools.partial(
        self.pallas_call,
        out_shape=out_shape,
        grid=grid,
        input_output_aliases={1: 0},
    )
    def reduce(x_ref, _, y_ref):
      i = pl.program_id(axis=0)
      if axis == 0:
        idx = (i, jnp.arange(n))
      else:
        idx = (jnp.arange(m), i)
      x = pl.load(x_ref, idx)
      pl.atomic_add(y_ref, (jnp.arange(y.shape[0]),), x)

    x = random.normal(random.key(0), (m, n))
    y = jnp.zeros(out_shape.shape, out_shape.dtype)
    y = reduce(x, y)
    y_ref = np.sum(x, axis=axis)
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)

  @parameterized.parameters(
      (0, 0, 1),
      (0, 1, 1),
      (1, 0, 1),
      (1, 1, 1),
      (2, 1, 1),
      (2, 1, 1),
  )
  def test_atomic_cas(self, init_value, cmp, new_value):
    @functools.partial(
        self.pallas_call, out_shape=(
          jax.ShapeDtypeStruct((), jnp.int32),
          jax.ShapeDtypeStruct((), jnp.int32)),
        input_output_aliases={0: 0})
    def swap(_, lock_ref, out_ref):
      out_ref[()] = pl.atomic_cas(lock_ref, cmp, new_value)

    lock, out = swap(init_value)
    np.testing.assert_allclose(lock, new_value if cmp == init_value else
                               init_value)
    np.testing.assert_allclose(out, init_value)

  @parameterized.parameters(1, 2, 3, 4, 8)
  def test_atomic_counter(self, num_threads):
    if self.INTERPRET:
      self.skipTest("While loop not supported in interpreter mode.")

    @functools.partial(
        self.pallas_call, out_shape=(
          jax.ShapeDtypeStruct((), jnp.int32),
          jax.ShapeDtypeStruct((), jnp.int32)),
        input_output_aliases={0: 0, 1: 1},
        grid=(num_threads,))
    def increment(_, __, lock_ref, counter_ref):
      def _cond(_):
        return pl.atomic_cas(lock_ref, 0, 1) == 1
      lax.while_loop(_cond, lambda a: a, 0)
      counter_ref[...] += 1
      pl.atomic_xchg(lock_ref, (), 0)

    lock, count = increment(0, 0)
    np.testing.assert_allclose(lock, 0)
    np.testing.assert_allclose(count, num_threads)

  @parameterized.parameters(False, True)
  def test_reduce_only_dim(self, use_store):
    m = 32
    x = random.normal(random.key(0), (m,), dtype=jnp.float32)
    out_shape = jax.ShapeDtypeStruct((), x.dtype)

    @functools.partial(
        self.pallas_call, out_shape=out_shape, grid=1, debug=False
    )
    def reduce(x_ref, y_ref):
      x = pl.load(x_ref, (jnp.arange(m),))
      y = jnp.sum(x, axis=-1)
      if use_store:
        pl.store(y_ref, (), y)
      else:
        y_ref[...] = y

    y = reduce(x)
    y_ref = jnp.sum(x, axis=-1)
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(*[
      (f"{op_name}_{dtype}_{axis}", op, dtype, axis)
      for op_name, op in [
          ("add", jnp.sum),
          ("max", jnp.max),
          ("min", jnp.min),
          ("argmax", jnp.argmax),
          ("argmin", jnp.argmin),
      ]
      for axis in [0, 1, (1,), (0, 1)]
      for dtype in ["float16", "float32", "int32", "uint32"]
      if isinstance(axis, int) or "arg" not in op_name
  ])
  def test_array_reduce(self, op, dtype, axis):
    m, n = 32, 8
    out_dtype = dtype
    if op in {jnp.argmin, jnp.argmax}:
      out_dtype = jnp.int32

    def make_x(key):
      if jnp.issubdtype(dtype, jnp.integer):
        return random.permutation(
            key, jnp.arange(m * n, dtype=dtype), independent=True
        ).reshape(m, n)
      else:
        return random.normal(key, (m, n), dtype=dtype)

    out_shape = jax.ShapeDtypeStruct(
        op(make_x(random.key(0)), axis=axis).shape, out_dtype
    )
    if isinstance(axis, int):
      grid = tuple(a for i, a in enumerate((m, n)) if i != axis)
    else:
      grid = tuple(a for i, a in enumerate((m, n)) if i not in axis)

    @functools.partial(self.pallas_call, out_shape=out_shape, grid=grid)
    def reduce(x_ref, y_ref):
      x = pl.load(x_ref, (jnp.arange(m)[:, None], jnp.arange(n)[None]))
      y = op(x, axis=axis)
      pl.store(y_ref, tuple(jnp.arange(d) for d in y.shape), y)

    for i, key in enumerate(random.split(random.key(0), 20)):
      x = make_x(key)
      y = reduce(x)
      y_ref = op(x, axis=axis)
      np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2, err_msg=i)

  @parameterized.product(
      axis=[0, 1],
      dtype=["float16", "float32", "int32", "uint32"],
  )
  def test_cumsum(self, dtype, axis):
    m, n = 32, 8
    out_dtype = dtype

    def make_x(key):
      if jnp.issubdtype(dtype, jnp.integer):
        return random.permutation(
            key, jnp.arange(m * n, dtype=dtype), independent=True
        ).reshape(m, n)
      else:
        return random.normal(key, (m, n), dtype=dtype)

    out_shape = jax.ShapeDtypeStruct((m, n), out_dtype)
    grid = ()

    @functools.partial(self.pallas_call, out_shape=out_shape, grid=grid)
    def reduce(x_ref, y_ref):
      x = x_ref[...]
      y_ref[...] = jnp.cumsum(x, axis=axis)

    for i, key in enumerate(random.split(random.key(0), 20)):
      x = make_x(key)
      y = reduce(x)
      y_ref = jnp.cumsum(x, axis=axis)
      np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2, err_msg=i)


class PallasOpsInterpretTest(PallasOpsTest):
  INTERPRET = True


class PallasPrimitivesTest(PallasTest):

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)), "<- a[:,:,:]"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)), "<- a[:3,:,:]"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)), "<- a[1:,:,:4]"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)), "<- a[b,:,:4]"),
    (lambda: (jnp.arange(5)[:, None], jnp.arange(3)[None], pl.ds(4)), "<- a[f,g,:4]"),
  ])
  def test_load_pretty_print(self, expr, expected):
    def body(x_ref):
      x = pl.load(x_ref, expr())
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)), "a[:,:,:] <-"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)), "a[:3,:,:] <-"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)), "a[1:,:,:4] <-"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)), "a[b,:,:4] <-"),
    (lambda: (jnp.arange(5)[:, None], jnp.arange(3)[None], pl.dslice(4)), "a[m,n,:4] <-"),
  ])
  def test_store_pretty_print(self, expr, expected):
    def body(x_ref):
      pl.store(x_ref, expr(), pl.load(x_ref, expr()))
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)),
     "c:i32[4,3,2], a[:,:,:] <-"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)),
     "c:i32[3,3,2], a[:3,:,:] <-"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)),
     "c:i32[3,3,4], a[1:,:,:4] <-"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)),
     "e:i32[5,3,4], a[b,:,:4] <-"),
    (lambda: (jnp.arange(5)[:, None], jnp.arange(3)[None], pl.dslice(4)),
     "o:i32[5,3,4], a[m,n,:4] <-"),
  ])
  def test_swap_pretty_print(self, expr, expected):
    def body(x_ref):
      x = pl.swap(x_ref, expr(), pl.load(x_ref, expr()))
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

class FusedAttentionTest(PallasTest):

  @parameterized.named_parameters(
      *[
          (
              (
                  f"{batch_size=}_{seq_len=}_{num_heads=}_{head_dim=}_{causal=}"
                  f"_{use_fwd=}_{use_segment_ids=}_{kwargs=}"
              ),
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              causal,
              use_fwd,
              use_segment_ids,
              kwargs,
          )
          for (
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              causal,
              use_fwd,
              use_segment_ids,
              kwargs,
          ) in [
              (1, 384, 1, 64, False, False, True, {}),
              (1, 384, 1, 64, False, False, False, {}),
              (2, 384, 2, 64, False, False, True, {}),
              (1, 384, 1, 64, True, False, True, {}),
              # (2, 384, 2, 64, True, False, True, {}), # TODO(sharadmv): Investigate.
              (1, 384, 8, 64, True, True, True, {}),
              (1, 384, 8, 64, True, True, False, {}),
              (2, 384, 8, 64, True, True, True, {}),
              # regression test: https://github.com/google/jax/pull/17314
              (1, 384, 8, 64, True, False, False, {'block_q': 128, 'block_k': 64}),
          ]
      ]
  )
  def test_fused_attention_fwd(
      self,
      batch_size,
      seq_len,
      num_heads,
      head_dim,
      causal,
      use_fwd,
      use_segment_ids,
      kwargs,
  ):
    k1, k2, k3 = random.split(random.key(0), 3)
    q = random.normal(
        k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    k = random.normal(
        k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    v = random.normal(
        k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    if use_segment_ids:
      segment_ids_1 = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
      segment_ids_2 = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
      segment_ids = jnp.concatenate((segment_ids_1, segment_ids_2), axis=-1)
    else:
      segment_ids = None

    if use_fwd:

      @jax.jit
      def impl(q, k, v):
        v, _ = jax.vjp(
            functools.partial(
                attention.mha, causal=causal, segment_ids=segment_ids, **kwargs
            ),
            q,
            k,
            v,
        )
        return v

    else:
      impl = functools.partial(
          attention.mha, causal=causal, segment_ids=segment_ids, **kwargs
      )
    o = impl(q, k, v)
    o_ref = attention.mha_reference(q, k, v, segment_ids, causal=causal)
    np.testing.assert_allclose(o, o_ref, atol=0.05)

  @parameterized.named_parameters(
      *[
          (
              (
                  f"{batch_size=}_{seq_len=}_{num_heads=}_{head_dim=}_{causal=}_"
                  f"{use_segment_ids=}"
              ),
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              causal,
              use_segment_ids,
          )
          for (
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              causal,
              use_segment_ids,
          ) in [
              (1, 384, 1, 32, False, True),
              (1, 384, 1, 32, False, False),
              (2, 384, 2, 32, False, True),
              (2, 384, 2, 32, False, False),
              # TODO(b/283035396): (1, 384, 1, 32, True, True),
              # TODO(b/283035396): (2, 384, 2, 32, True, True),
          ]
      ]
  )
  def test_fused_attention_bwd(
      self, batch_size, seq_len, num_heads, head_dim, causal, use_segment_ids
  ):
    k1, k2, k3 = random.split(random.key(0), 3)
    q = random.normal(
        k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    k = random.normal(
        k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    v = random.normal(
        k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    if use_segment_ids:
      segment_ids_1 = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
      segment_ids_2 = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
      segment_ids = jnp.concatenate((segment_ids_1, segment_ids_2), axis=-1)
    else:
      segment_ids = None

    def f(q, k, v):
      return attention.mha(q, k, v, segment_ids, causal=causal).sum()

    def f_ref(q, k, v):
      return attention.mha_reference(q, k, v, segment_ids, causal=causal).sum()

    dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
    dq_ref, dk_ref, dv_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)
    # TODO(sharadmv): Fix test.
    np.testing.assert_allclose(dq, dq_ref, atol=0.14)
    np.testing.assert_allclose(dk, dk_ref, atol=0.14)
    np.testing.assert_allclose(dv, dv_ref, atol=0.05)


class FusedAttentionInterpreterTest(PallasTest):
  INTERPRET = True

class FusedLayerNormTest(PallasTest):

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_fused_layernorm_fwd(self, batch_size, seq_len, embed_dim):
    k1, k2, k3 = random.split(random.key(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    o = layer_norm.layer_norm(x, w, b)
    o_ref = layer_norm.layer_norm_reference(x, w, b)
    np.testing.assert_allclose(o, o_ref, atol=1e-5)

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_fused_layernorm_bwd(self, batch_size, seq_len, embed_dim):
    k1, k2, k3 = random.split(random.key(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    def f(x, w, b):
      return layer_norm.layer_norm(x, w, b).sum()

    def f_ref(x, w, b):
      return layer_norm.layer_norm_reference(x, w, b).sum()

    dx, dw, db = jax.grad(f, argnums=(0, 1, 2))(x, w, b)
    dx_ref, dw_ref, db_ref = jax.grad(f_ref, argnums=(0, 1, 2))(x, w, b)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(db, db_ref, rtol=1e-2, atol=1e-2)


class FusedLayerNormInterpreterTest(PallasTest):
  INTERPRET = True


class RmsNormTest(PallasTest):

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_rms_fwd(self, batch_size, seq_len, embed_dim):
    k1, k2, k3 = random.split(random.key(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    o = rms_norm.rms_norm(x, w, b)
    o_ref = rms_norm.rms_norm_reference(x, w, b)
    np.testing.assert_allclose(o, o_ref, atol=1e-5)

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_rms_norm_bwd(self, batch_size, seq_len, embed_dim):
    k1, k2, k3 = random.split(random.key(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    def f(x, w, b):
      return rms_norm.rms_norm(x, w, b).sum()

    def f_ref(x, w, b):
      return rms_norm.rms_norm_reference(x, w, b).sum()

    dx, dw, db = jax.grad(f, argnums=(0, 1, 2))(x, w, b)
    dx_ref, dw_ref, db_ref = jax.grad(f_ref, argnums=(0, 1, 2))(x, w, b)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(db, db_ref, rtol=1e-2, atol=1e-2)

class RmsNormInterpreterTest(PallasTest):
  INTERPRET = True

class SoftmaxTest(PallasTest):

  @parameterized.product(
      shape=[(1024, 125), (4, 1024, 125)],
      dtype=[jnp.bfloat16, jnp.float16, jnp.float32]
  )
  def test_softmax(self, shape, dtype):
    x = jax.random.normal(random.key(0), shape, dtype=dtype)

    atol, rtol = {
        jnp.bfloat16: (1e-2, 1e-4),
        jnp.float16: (1e-2, 1e-4),
        jnp.float32: (1e-7, 1e-6),
    }[dtype]

    # We upcast to float32 because NumPy <2.0 does not handle custom dtypes
    # properly. See https://github.com/google/jax/issues/11014.
    np.testing.assert_allclose(
        softmax.softmax(x, axis=-1).astype(jnp.float32),
        jax.nn.softmax(x, axis=-1).astype(jnp.float32),
        atol=atol,
        rtol=rtol,
    )


class SoftmaxInterpreterTest(PallasTest):
  INTERPRET = True


class PallasInterpretModeOutOfBoundsTest(PallasTest):

  INTERPRET: bool = True

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
        pl.BlockSpec(lambda i, j, k: (i, k), (block_size, block_size)),
        pl.BlockSpec(lambda i, j, k: (k, j), (block_size, block_size)),
    ]
    out_spec = pl.BlockSpec(lambda i, j, k: (i, j), (block_size, block_size))

    def _unmasked_matmul_kernel(x_ref, y_ref, o_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        o_ref[...] = jnp.zeros_like(o_ref)

      o_ref[...] += x_ref[...] @ y_ref[...]

    out = pl.pallas_call(
        _unmasked_matmul_kernel,
        out_shape=expected,
        grid=(1, 1, 2),
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=True,
    )(x, y)

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

    out = pl.pallas_call(
        _masked_matmul_kernel,
        out_shape=expected,
        grid=(1, 1, 2),
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=True,
    )(x, y)

    # TODO(justinfu): This test has low precision on GPU. Improve precision.
    if jtu.test_device_matches(["gpu"]):
      atol = 1e-2
    else:
      atol = 1e-5

    # With a masked matmul implementation, uninitialized values will be
    # masked before computation. This should return the correct result.
    with self.subTest('MaskedOutputIsCorrect'):
      np.testing.assert_allclose(out, expected, atol=atol)


class PallasCheckifyTest(PallasTest):
  # TODO(b/346651778): Support non-interpret mode checkify.
  INTERPRET: bool = True

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
    in_specs = [pl.BlockSpec(lambda x : (x,), (1,))]
    out_shape = jax.ShapeDtypeStruct((1,), dtype=jnp.float32)
    pallas_call = self.pallas_call(kernel,
                                 grid=(num_loops,),
                                 in_specs=in_specs,
                                 out_shape=out_shape)

    checked_call = checkify.checkify(pallas_call,
                                     errors=checkify.all_checks)
    err, _ = checked_call(input_arr)
    with self.assertRaisesRegex(
        checkify.JaxRuntimeError, f"failed on loop {fail_iteration}"):
      err.throw()


if __name__ == "__main__":
  absltest.main()
