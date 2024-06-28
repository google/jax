# Copyright 2019 The JAX Authors.
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


import itertools

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
from jax import numpy as jnp
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.numpy.util import promote_dtypes_complex
from jax._src.numpy.fft import _fft_norm

config.parse_flags_with_absl()

FFT_NORMS = [None, "ortho", "forward", "backward"]


float_dtypes = jtu.dtypes.floating
inexact_dtypes = jtu.dtypes.inexact
real_dtypes = float_dtypes + jtu.dtypes.integer + jtu.dtypes.boolean
all_dtypes = real_dtypes + jtu.dtypes.complex


def _get_fftn_test_axes(shape):
  axes = [[]]
  ndims = len(shape)
  # XLA's FFT op only supports up to 3 innermost dimensions.
  if ndims <= 3:
    axes.append(None)
  for naxes in range(1, min(ndims, 3) + 1):
    axes.extend(itertools.combinations(range(ndims), naxes))
  for index in range(1, ndims + 1):
    axes.append((-index,))
  return axes

def _get_fftn_test_s(shape, axes):
  s_list = [None]
  if axes is not None:
    s_list.extend(itertools.product(*[[shape[ax]+i for i in range(-shape[ax]+1, shape[ax]+1)] for ax in axes]))
  return s_list

def _get_fftn_func(module, inverse, real):
  if inverse:
    return _irfft_with_zeroed_inputs(module.irfftn) if real else module.ifftn
  else:
    return module.rfftn if real else module.fftn


def _irfft_with_zeroed_inputs(irfft_fun):
  # irfft isn't defined on the full domain of inputs, so in order to have a
  # well defined derivative on the whole domain of the function, we zero-out
  # the imaginary part of the first and possibly the last elements.
  def wrapper(z, axes, s=None, norm=None):
    return irfft_fun(_zero_for_irfft(z, axes), axes=axes, s=s, norm=norm)
  return wrapper


def _zero_for_irfft(z, axes):
  if axes is not None and not axes:
    return z
  axis = z.ndim - 1 if axes is None else axes[-1]
  try:
    size = z.shape[axis]
  except IndexError:
    return z  # only if axis is invalid, as occurs in some tests
  if size % 2:
    parts = [lax.slice_in_dim(z.real, 0, 1, axis=axis).real,
             lax.slice_in_dim(z.real, 1, size - 1, axis=axis),
             lax.slice_in_dim(z.real, size - 1, size, axis=axis).real]
  else:
    parts = [lax.slice_in_dim(z.real, 0, 1, axis=axis).real,
             lax.slice_in_dim(z.real, 1, size, axis=axis)]
  return jnp.concatenate(parts, axis=axis)


class FftTest(jtu.JaxTestCase):

  def testLaxFftAcceptsStringTypes(self):
    rng = jtu.rand_default(self.rng())
    x = rng((10,), np.complex64)
    self.assertAllClose(np.fft.fft(x).astype(np.complex64),
                        lax.fft(x, "FFT", fft_lengths=(10,)))
    self.assertAllClose(np.fft.fft(x).astype(np.complex64),
                        lax.fft(x, "fft", fft_lengths=(10,)))

  def testLaxFftErrors(self):
    with self.assertRaisesRegex(
        ValueError,
        r"FFT input shape \(14, 15\) must have at least as many input "
        r"dimensions as fft_lengths \(4, 5, 6\)"):
      lax.fft(np.ones((14, 15)), fft_type="fft", fft_lengths=(4, 5, 6))
    with self.assertRaisesRegex(
        ValueError,
        r"FFT input shape \(14, 15\) minor dimensions must be equal to "
        r"fft_lengths \(17,\)"):
      lax.fft(np.ones((14, 15)), fft_type="fft", fft_lengths=(17,))
    with self.assertRaisesRegex(
        ValueError,
        r"RFFT input shape \(2, 14, 15\) minor dimensions must be equal to "
        r"fft_lengths \(14, 12\)"):
      lax.fft(np.ones((2, 14, 15)), fft_type="rfft", fft_lengths=(14, 12))
    with self.assertRaisesRegex(
        ValueError,
        r"IRFFT input shape \(2, 14, 15\) minor dimensions must be equal to "
        r"all except the last fft_length, got fft_lengths=\(13, 15\)"):
      lax.fft(np.ones((2, 14, 15)), fft_type="irfft", fft_lengths=(13, 15))
    with self.assertRaisesRegex(
        ValueError, "RFFT input must be float32 or float64, got bfloat16"):
      lax.fft(np.ones((14, 15), jnp.bfloat16), fft_type="rfft",
              fft_lengths=(5, 6))

  @parameterized.parameters((np.float32,), (np.float64,))
  def testLaxIrfftDoesNotMutateInputs(self, dtype):
    if dtype == np.float64 and not config.enable_x64.value:
      raise self.skipTest("float64 requires jax_enable_x64=true")
    x = (1 + 1j) * jnp.array([[1.0, 2.0], [3.0, 4.0]],
                             dtype=dtypes.to_complex_dtype(dtype))
    y = np.asarray(jnp.fft.irfft2(x))
    z = np.asarray(jnp.fft.irfft2(x))
    self.assertAllClose(y, z)

  @jtu.sample_product(
    [dict(inverse=inverse, real=real, dtype=dtype)
     for inverse in [False, True]
     for real in [False, True]
     for dtype in (real_dtypes if real and not inverse else all_dtypes)
    ],
    [dict(shape=shape, axes=axes, s=s)
     for shape in [(10,), (10, 10), (9,), (2, 3, 4), (2, 3, 4, 5)]
     for axes in _get_fftn_test_axes(shape)
     for s in _get_fftn_test_s(shape, axes)
    ],
    norm=FFT_NORMS,
  )
  def testFftn(self, inverse, real, shape, dtype, axes, s, norm):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_op = _get_fftn_func(jnp.fft, inverse, real)
    np_op = _get_fftn_func(np.fft, inverse, real)
    jnp_fn = lambda a: jnp_op(a, axes=axes, norm=norm)
    np_fn = lambda a: np_op(a, axes=axes, norm=norm) if axes is None or axes else a
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker)
    # Test gradient for differentiable types.
    if (config.enable_x64.value and
        dtype in (float_dtypes if real and not inverse else inexact_dtypes)):
      # TODO(skye): can we be more precise?
      tol = 0.15
      jtu.check_grads(jnp_fn, args_maker(), order=2, atol=tol, rtol=tol)

    # check dtypes
    dtype = jnp_fn(rng(shape, dtype)).dtype
    expected_dtype = jnp.promote_types(float if inverse and real else complex, dtype)
    self.assertEqual(dtype, expected_dtype)

  def testIrfftTranspose(self):
    # regression test for https://github.com/google/jax/issues/6223
    def build_matrix(linear_func, size):
      return jax.vmap(linear_func)(jnp.eye(size, size))

    def func(x):
      x, = promote_dtypes_complex(x)
      return jnp.fft.irfft(jnp.concatenate([jnp.zeros_like(x, shape=1),
                                            x[:2] + 1j*x[2:]]))

    def func_transpose(x):
      return jax.linear_transpose(func, x)(x)[0]

    matrix = build_matrix(func, 4)
    matrix2 = build_matrix(func_transpose, 4).T
    self.assertAllClose(matrix, matrix2)

  @jtu.sample_product(
    inverse=[False, True],
    real=[False, True],
  )
  def testFftnErrors(self, inverse, real):
    rng = jtu.rand_default(self.rng())
    name = 'fftn'
    if real:
      name = 'r' + name
    if inverse:
      name = 'i' + name
    func = _get_fftn_func(jnp.fft, inverse, real)
    self.assertRaisesRegex(
        ValueError,
        "jax.numpy.fft.{} only supports 1D, 2D, and 3D FFTs. "
        "Got axes None with input rank 4.".format(name),
        lambda: func(rng([2, 3, 4, 5], dtype=np.float64), axes=None))
    self.assertRaisesRegex(
        ValueError,
        f"jax.numpy.fft.{name} does not support repeated axes. Got axes \\[1, 1\\].",
        lambda: func(rng([2, 3], dtype=np.float64), axes=[1, 1]))
    self.assertRaises(
        ValueError, lambda: func(rng([2, 3], dtype=np.float64), axes=[2]))
    self.assertRaises(
        ValueError, lambda: func(rng([2, 3], dtype=np.float64), axes=[-3]))

  def testFftEmpty(self):
    out = jnp.fft.fft(jnp.zeros((0,), jnp.complex64)).block_until_ready()
    self.assertArraysEqual(jnp.zeros((0,), jnp.complex64), out)

  @jtu.sample_product(
    [dict(inverse=inverse, real=real, hermitian=hermitian, dtype=dtype)
      for inverse in [False, True]
      for real in [False, True]
      for hermitian in [False, True]
      for dtype in (real_dtypes if (real and not inverse) or (hermitian and inverse)
                                else all_dtypes)
    ],
    shape=[(10,)],
    n=[None, 1, 7, 13, 20],
    axis=[-1, 0],
  )
  def testFft(self, inverse, real, hermitian, shape, dtype, n, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    name = 'fft'
    if real:
      name = 'r' + name
    elif hermitian:
      name = 'h' + name
    if inverse:
      name = 'i' + name
    jnp_op = getattr(jnp.fft, name)
    np_op = getattr(np.fft, name)
    jnp_fn = lambda a: jnp_op(a, n=n, axis=axis)
    np_fn = lambda a: np_op(a, n=n, axis=axis)
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    inverse=[False, True],
    real=[False, True],
    hermitian=[False, True],
  )
  def testFftErrors(self, inverse, real, hermitian):
    rng = jtu.rand_default(self.rng())
    name = 'fft'
    if real:
      name = 'r' + name
    elif hermitian:
      name = 'h' + name
    if inverse:
      name = 'i' + name
    func = getattr(jnp.fft, name)

    self.assertRaisesRegex(
      ValueError,
      f"jax.numpy.fft.{name} does not support multiple axes. "
      f"Please use jax.numpy.fft.{name}n. Got axis = \\[1, 1\\].",
      lambda: func(rng([2, 3], dtype=np.float64), axis=[1, 1])
    )
    self.assertRaisesRegex(
      ValueError,
      f"jax.numpy.fft.{name} does not support multiple axes. "
      f"Please use jax.numpy.fft.{name}n. Got axis = \\(1, 1\\).",
      lambda: func(rng([2, 3], dtype=np.float64), axis=(1, 1))
    )
    self.assertRaises(
        ValueError, lambda: func(rng([2, 3], dtype=np.float64), axis=[2]))
    self.assertRaises(
        ValueError, lambda: func(rng([2, 3], dtype=np.float64), axis=[-3]))

  @jtu.sample_product(
    [dict(inverse=inverse, real=real, dtype=dtype)
     for inverse in [False, True]
     for real in [False, True]
     for dtype in (real_dtypes if real and not inverse else all_dtypes)
    ],
    shape=[(16, 8, 4, 8), (16, 8, 4, 8, 4)],
    axes=[(-2, -1), (0, 1), (1, 3), (-1, 2)],
    norm=FFT_NORMS,
  )
  def testFft2_(self, inverse, real, shape, dtype, axes, norm):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    name = 'fft2'
    if real:
      name = 'r' + name
    if inverse:
      name = 'i' + name
    jnp_op = getattr(jnp.fft, name)
    np_op = getattr(np.fft, name)
    jnp_fn = lambda a: jnp_op(a, axes=axes, norm=norm)
    np_fn = lambda a: np_op(a, axes=axes, norm=norm) if axes is None or axes else a
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    inverse=[False, True],
    real=[False, True],
  )
  def testFft2Errors(self, inverse, real):
    rng = jtu.rand_default(self.rng())
    name = 'fft2'
    if real:
      name = 'r' + name
    if inverse:
      name = 'i' + name
    func = getattr(jnp.fft, name)

    self.assertRaisesRegex(
      ValueError,
      "jax.numpy.fft.{} only supports 2 axes. "
      "Got axes = \\[0\\].".format(name),
      lambda: func(rng([2, 3], dtype=np.float64), axes=[0])
    )
    self.assertRaisesRegex(
      ValueError,
      "jax.numpy.fft.{} only supports 2 axes. "
      "Got axes = \\(0, 1, 2\\).".format(name),
      lambda: func(rng([2, 3, 3], dtype=np.float64), axes=(0, 1, 2))
    )
    self.assertRaises(
      ValueError, lambda: func(rng([2, 3], dtype=np.float64), axes=[2, 3]))
    self.assertRaises(
      ValueError, lambda: func(rng([2, 3], dtype=np.float64), axes=[-3, -4]))

  @jtu.sample_product(
    dtype=all_dtypes,
    size=[9, 10, 101, 102],
    d=[0.1, 2.],
    device=[None, -1],
  )
  def testFftfreq(self, size, d, dtype, device):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng([size], dtype),)
    jnp_op = jnp.fft.fftfreq
    np_op = np.fft.fftfreq
    if device is not None:
      device = jax.devices()[device]
    jnp_fn = lambda a: jnp_op(size, d=d, device=device)
    np_fn = lambda a: np_op(size, d=d)
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker)
    # Test gradient for differentiable types.
    if dtype in inexact_dtypes:
      tol = 0.15  # TODO(skye): can we be more precise?
      jtu.check_grads(jnp_fn, args_maker(), order=2, atol=tol, rtol=tol)
    # Test device
    if device is not None:
      out = jnp_fn(args_maker())
      self.assertEqual(out.devices(), {device})

  @jtu.sample_product(n=[[0, 1, 2]])
  def testFftfreqErrors(self, n):
    name = 'fftfreq'
    func = jnp.fft.fftfreq
    self.assertRaisesRegex(
      ValueError,
      "The n argument of jax.numpy.fft.{} only takes an int. "
      "Got n = \\[0, 1, 2\\].".format(name),
      lambda: func(n=n)
    )
    self.assertRaisesRegex(
      ValueError,
      "The d argument of jax.numpy.fft.{} only takes a single value. "
      "Got d = \\[0, 1, 2\\].".format(name),
      lambda: func(n=10, d=n)
    )

  @jtu.sample_product(
    dtype=all_dtypes,
    size=[9, 10, 101, 102],
    d=[0.1, 2.],
    device=[None, -1],
  )
  def testRfftfreq(self, size, d, dtype, device):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng([size], dtype),)
    jnp_op = jnp.fft.rfftfreq
    np_op = np.fft.rfftfreq
    if device is not None:
      device = jax.devices()[device]
    jnp_fn = lambda a: jnp_op(size, d=d, device=device)
    np_fn = lambda a: np_op(size, d=d)
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker)
    # Test gradient for differentiable types.
    if dtype in inexact_dtypes:
      tol = 0.15  # TODO(skye): can we be more precise?
      jtu.check_grads(jnp_fn, args_maker(), order=2, atol=tol, rtol=tol)
    # Test device
    if device is not None:
      out = jnp_fn(args_maker())
      self.assertEqual(out.devices(), {device})

  @jtu.sample_product(n=[[0, 1, 2]])
  def testRfftfreqErrors(self, n):
    name = 'rfftfreq'
    func = jnp.fft.rfftfreq
    self.assertRaisesRegex(
      ValueError,
      "The n argument of jax.numpy.fft.{} only takes an int. "
      "Got n = \\[0, 1, 2\\].".format(name),
      lambda: func(n=n)
    )
    self.assertRaisesRegex(
      ValueError,
      "The d argument of jax.numpy.fft.{} only takes a single value. "
      "Got d = \\[0, 1, 2\\].".format(name),
      lambda: func(n=10, d=n)
    )

  @jtu.sample_product(
    [dict(shape=shape, axes=axes)
     for shape in [[9], [10], [101], [102], [3, 5], [3, 17], [5, 7, 11]]
     for axes in _get_fftn_test_axes(shape)
    ],
    dtype=all_dtypes,
  )
  def testFftshift(self, shape, dtype, axes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda arg: jnp.fft.fftshift(arg, axes=axes)
    np_fn = lambda arg: np.fft.fftshift(arg, axes=axes)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axes=axes)
     for shape in [[9], [10], [101], [102], [3, 5], [3, 17], [5, 7, 11]]
     for axes in _get_fftn_test_axes(shape)
    ],
    dtype=all_dtypes,
  )
  def testIfftshift(self, shape, dtype, axes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda arg: jnp.fft.ifftshift(arg, axes=axes)
    np_fn = lambda arg: np.fft.ifftshift(arg, axes=axes)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker)

  @jtu.sample_product(
    norm=["ortho", "forward"],
    func_name = ["fft", "ifft"],
    dtype=jtu.dtypes.integer
  )
  def testFftnormOverflow(self, norm, func_name, dtype):
    # non-regression test for gh-18453

    shape = jnp.array([3] + [900] * 3, dtype=dtype)
    jax_norm = _fft_norm(shape, func_name, norm)
    np_norm = np.array(shape).prod(dtype=np.float64)
    if norm == "ortho":
      np_norm = np.sqrt(np_norm)
    if func_name[0] != "i":
      np_norm = np.reciprocal(np_norm)
    self.assertArraysAllClose(jax_norm, np_norm, rtol=3e-8, check_dtypes=False)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
