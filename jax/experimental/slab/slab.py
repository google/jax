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

from collections.abc import Iterable, Sequence
from functools import partial, reduce
import sys
import typing
from typing import NamedTuple, Union

import numpy as np

import jax
import jax.numpy as jnp

from jax._src import core
from jax._src import util

map, zip = util.safe_map, util.safe_zip

DInt = jax.Array
Address = DInt
XInt = Union[int, DInt]
DShape = tuple[XInt, ...]
SShape = tuple[int, ...]
DType = jnp.dtype

class Slab(NamedTuple):
  data: jax.Array
  cursor: Address

@jax.tree_util.register_pytree_node_class
class SlabView(NamedTuple):
  addr: Address
  shape: DShape
  dtype: DType

  def size(self):
    return jnp.prod(jnp.array(self.shape))

  def ndim(self):
    return len(self.shape)

  def tree_flatten(self):
    return (self.addr, self.shape), self.dtype

  @classmethod
  def tree_unflatten(cls, dtype, xs):
    addr, shape = xs
    return cls(addr, shape, dtype)

word_b = 4
phrase_b = 512
phrase_w = 128
tile_aspect = 8

def xceil_div(x: XInt, y: XInt) -> XInt:
  """ceil(x / y)"""
  return (x + y - 1) // y

def _xadd(x: XInt, y: XInt) -> XInt:
  return x + y

def _xmul(x: XInt, y: XInt) -> XInt:
  return x * y

def xadd(*xs: XInt) -> XInt:
  return reduce(_xadd, xs, typing.cast(XInt, 0))

def xmul(*xs: XInt) -> XInt:
  return reduce(_xmul, xs, typing.cast(XInt, 1))

def xsum(xs: Iterable[XInt]) -> XInt:
  return xadd(*list(xs))

def xprod(xs: Iterable[XInt]) -> XInt:
  return xmul(*list(xs))

def static_int(x: XInt) -> bool:
  return isinstance(core.get_aval(x), core.ConcreteArray)

def static_shape(s: DShape) -> bool:
  return all(map(static_int, s))

def assert_static_int(x: XInt) -> int:
  if not static_int(x):
    raise TypeError(f'{x} is not a static int')
  return int(x)

def assert_static_shape(s: DShape) -> SShape:
  if not static_shape(s):
    raise TypeError(f'{s} is not a static shape')
  return tuple(map(int, s))

def tile_shape(shape: DShape, dtype) -> SShape:
  # Units: (1, 1, ..., elements, 1)
  if len(shape) < 2:
    raise NotImplementedError('matrices or bust')
  num_leading = len(shape) - 2
  return (1,) * num_leading + (tile_aspect * word_b // dtype.itemsize,
                               phrase_b // word_b)

def tile_phrases(shape: DShape, dtype: DType):
  # Units: phrases
  return xprod(tile_shape(shape, dtype)) * dtype.itemsize // phrase_b

def slab_make(num_phrases):
  return Slab(jnp.zeros((num_phrases, phrase_w), dtype=jnp.uint32),
              jnp.array(0, dtype=jnp.int32))

def slab_alloc(slab: Slab, shape: DShape, dtype):
  if len(shape) < 2:
    raise NotImplementedError('matrices or bust')
  tiled_shape = map(xceil_div, shape, tile_shape(shape, dtype))
  num_p = xmul(*tiled_shape, tile_phrases(shape, dtype))
  new_slab = Slab(slab.data, slab.cursor + num_p)
  slab_val = SlabView(slab.cursor, shape, dtype)
  return new_slab, slab_val

def strides(xs):
  s = 1
  ss = []
  for x in reversed(xs):
    ss.append(s)
    s *= x
  return tuple(reversed(ss))

def slab_slices(view, slice_base_e: DShape, slice_shape_e: SShape):
  view_shape_e = tile_shape(view.shape, view.dtype)
  # dassert all(s % t == 0 for s, t in zip(slice_base,  view_shape_e))
  # dassert all(s % t == 0 for s, t in zip(slice_shape, view_shape_e))
  slice_base_t  = [s // t for s, t in zip(slice_base_e,  view_shape_e)]
  slice_shape_t = [s // t for s, t in zip(slice_shape_e, view_shape_e)]
  tiled_shape = map(xceil_div, view.shape, view_shape_e)
  tiled_strides = strides(tiled_shape)
  tp = tile_phrases(view.shape, view.dtype)
  for idx in np.ndindex(*slice_shape_t[:-1]):
    linear_idx_t = xsum(
        map(xmul, map(xadd, slice_base_t, (*idx, 0)), tiled_strides))
    yield (view.addr + linear_idx_t * tp, slice_shape_t[-1] * tp)

def reinterpret_cast(x: jax.Array, shape: SShape, dtype: DType):
  x_bytes = x.size * x.dtype.itemsize
  if -1 in shape:
    assert x_bytes % xprod(s for s in shape if s != -1) * dtype.itemsize == 0
  else:
    assert x_bytes == xprod(shape) * dtype.itemsize, (x.shape, x.dtype, shape, dtype)
  if x.dtype.itemsize != dtype.itemsize:
    # reshape(x, -1) in conversion below becomes reshape(-1, a, b) for some a,b
    raise NotImplementedError('todo')
  return jax.lax.bitcast_convert_type(x.reshape(-1), dtype).reshape(shape)

def slab_read(slab, view, slice_base: DShape, slice_shape: SShape):
  view_tile_shape = tile_shape(view.shape, view.dtype)
  tiled_shape = assert_static_shape(
      tuple(map(xceil_div, slice_shape, view_tile_shape)))
  slices = [
      jax.lax.dynamic_slice_in_dim(slab.data, addr, phrases)
      for addr, phrases in slab_slices(view, slice_base, slice_shape)]
  slice_mem = jnp.stack(slices, axis=0)
  return reinterpret_cast(
      slice_mem, (*tiled_shape, *view_tile_shape), view.dtype
      ).swapaxes(-2, -3).reshape(slice_shape)

# TODO: just take vjp of slab_read
def slab_write(slab, view, slice_base: DShape, inval: jax.Array):
  slice_shape = inval.shape
  view_tile_shape = tile_shape(view.shape, view.dtype)
  tiled_shape = map(xceil_div, inval.shape, view_tile_shape)
  inval_linearized = inval.reshape(
      *tiled_shape[:-1], view_tile_shape[-2], tiled_shape[-1], view_tile_shape[-1]
      ).swapaxes(-2, -3)
  slice_mem = reinterpret_cast(inval_linearized, (-1, phrase_w),
                               jnp.dtype('uint32'))
  slice_addr = 0
  new_slab = slab.data
  for slab_addr, slice_sz_p in slab_slices(view, slice_base, slice_shape):
    s = jax.lax.dynamic_slice_in_dim(slice_mem, slice_addr, slice_sz_p)
    slice_addr += slice_sz_p
    new_slab = jax.lax.dynamic_update_slice_in_dim(
        new_slab, s, slab_addr, axis=0)
  return Slab(new_slab, slab.cursor)

def elementwise(f, slab: Slab, xs: Sequence[SlabView], out: SlabView):
  if len(xs) == 0:
    raise TypeError('missing input arguments')
  x = xs[0]
  for y in xs[1:]:
    if x.shape != y.shape:
      raise ValueError(f'elementwise shapes mismatch: {x.shape} != {y.shape}')
    if x.dtype != y.dtype:
      raise ValueError(f'elementwise dtypes mismatch: {x.dtype} != {y.dtype}')
  if x.shape != out.shape:
    raise ValueError(
        f'elementwise input/output shape mismatch: {x.shape} != {out.shape}')

  tiled_shape = map(xceil_div, x.shape, tile_shape(x.shape, x.dtype))
  x_sz_p = xprod(tiled_shape) * tile_phrases(x.shape, x.dtype)
  compute_tile_p = 16
  num_whole_blocks = x_sz_p // compute_tile_p

  def f_u32(*zs):
    a = zs[0]
    return reinterpret_cast(
        f(*[reinterpret_cast(z, a.shape, x.dtype) for z in zs]),
        a.shape, jnp.dtype('uint32'))

  def body(i_b, mem):
    i_p = i_b * compute_tile_p
    slices = [
        jax.lax.dynamic_slice_in_dim(mem, z.addr + i_p, compute_tile_p)
        for z in xs]
    out_slice = f_u32(*slices)
    return jax.lax.dynamic_update_slice_in_dim(
        mem, out_slice, out.addr + i_p, axis=0)
  mem = jax.lax.fori_loop(0, num_whole_blocks, body, slab.data)

  epi_start_p = num_whole_blocks * compute_tile_p
  epi_size_p = x_sz_p - epi_start_p
  slices = [
      jax.lax.dynamic_slice_in_dim(mem, z.addr + epi_start_p, compute_tile_p)
      for z in xs]
  out_slice = f_u32(*slices)
  return Slab(masked_store(mem, out.addr + epi_start_p, out_slice, epi_size_p),
              slab.cursor)

def masked_store(mem, addr, update, num_p):
  update_p = update.shape[0]
  prev_val = jax.lax.dynamic_slice_in_dim(mem, addr, update_p)
  new_val = jnp.where(jnp.arange(update_p)[:, None] < num_p, update, prev_val)
  return jax.lax.dynamic_update_slice_in_dim(mem, new_val, addr, axis=0)

def _matmul(slab: Slab, ins: Sequence[SlabView], out: SlabView):
  lhs, rhs = ins
  dtype = lhs.dtype
  n, k, m = (*lhs.shape, rhs.shape[1])
  # todo: shape + dtype check
  # dassert shapes are tile aligned
  tile_n, tile_k, tile_m = 128, 128, 128
  n_tiles = n // tile_n
  k_tiles = k // tile_k
  m_tiles = m // tile_m

  mem = slab
  def loop_n(ni, mem):
    def loop_m(mi, mem):
      acc = jnp.zeros((tile_n, tile_m), dtype=dtype)
      def loop_k(ki, acc):
        lhs_tile = slab_read(mem, lhs, (ni * tile_n, ki * tile_k), (tile_n, tile_k))
        rhs_tile = slab_read(mem, rhs, (ki * tile_k, mi * tile_m), (tile_k, tile_m))
        acc += lhs_tile @ rhs_tile
        return acc
      acc = jax.lax.fori_loop(0, k_tiles, loop_k, acc)
      return slab_write(mem, out, (ni * tile_n, mi * tile_m), acc)
    return jax.lax.fori_loop(0, m_tiles, loop_m, mem)
  mem = jax.lax.fori_loop(0, n_tiles, loop_n, mem)
  return mem

def make_allocating_op(op, type_rule):
  def made_op(slab, *xs: SlabView):
    out_shape, out_dtype = type_rule(*xs)
    slab, out = slab_alloc(slab, out_shape, out_dtype)
    slab = op(slab, xs, out)
    return slab, out
  return made_op

add = make_allocating_op(partial(elementwise, jax.lax.add),
                         lambda x, *_: (x.shape, x.dtype))
mul = make_allocating_op(partial(elementwise, jax.lax.mul),
                         lambda x, *_: (x.shape, x.dtype))
tanh = make_allocating_op(partial(elementwise, jax.lax.tanh),
                          lambda x, *_: (x.shape, x.dtype))
matmul = make_allocating_op(_matmul,
                            lambda a, b: ((a.shape[0], b.shape[1]), a.dtype))

def parse_arr(i, s):
  shape = eval(s)
  return np.random.RandomState(i).normal(size=shape).astype(np.float32)

def print_seg(msg):
  print()
  print(f'-- {msg}')
  print()

def make_jaxpr_slab_write(slab, view, inval):
  return jax.make_jaxpr(
      lambda slab, x: slab_write(slab, view, (0, 0), x))(slab, inval)

def make_jaxpr_slab_read(slab, view, outval_shape):
  return jax.make_jaxpr(
      lambda slab: slab_read(slab, view, (0, 0), outval_shape))(slab)

def slab_download(slab, v):
  if not static_shape(v.shape): raise Exception
  return slab_read(slab, v, (0,) * v.ndim(), v.shape)

def slab_upload(slab, x):
  slab, xv = slab_alloc(slab, x.shape, x.dtype)
  slab = slab_write(slab, xv, (0,) * x.ndim, x)
  return slab, xv

def chain(slab, fs, *argss, unary=False):
  if callable(fs):
    fs = [fs] * len(argss)
  outss = []
  for f, args in zip(fs, argss):
    if unary:
      slab, outs = f(slab, args)
    else:
      slab, outs = f(slab, *args)
    outss.append(outs)
  return slab, outss

def test_binop(op, ref_op, slab, x, y):
  z = ref_op(x, y)
  slab, xv = slab_upload(slab, x)
  slab, yv = slab_upload(slab, y)
  slab, zv = op(slab, xv, yv)
  assert jnp.allclose(slab_download(slab, xv), x, atol=1e-4)
  assert jnp.allclose(slab_download(slab, yv), y, atol=1e-4)
  assert jnp.allclose(slab_download(slab, zv), z, atol=1e-4)

def main(args):
  xs = map(parse_arr, range(len(args)), args)
  assert all(len(x.shape) == 2 for x in xs)

  slab = slab_make(1024)

  x, y, *_ = xs
  test_binop(add, jax.lax.add, slab, x, x)
  test_binop(mul, jax.lax.mul, slab, x, x)
  test_binop(matmul, lambda a, b: a @ b, slab, x, y)

  def put(slab, x):
    slab, v = slab_upload(slab, x)
    print_seg('slab_read result')
    print(slab_download(slab, v))
    return slab, v

  slab, vals = chain(slab, put, *xs, unary=True)

  if len(vals) >= 2:
    x, y, *_ = vals
    slab, z = mul(slab, x, x)
    print_seg('mul')
    print(slab_download(slab, z))
    slab, w = add(slab, x, z)
    print_seg('add')
    print(slab_download(slab, w))


if __name__ == '__main__':
  main(sys.argv[1:])
