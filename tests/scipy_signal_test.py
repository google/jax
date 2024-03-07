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


from functools import partial
import unittest

from absl.testing import absltest

import numpy as np
import scipy.signal as osp_signal

import jax
from jax import lax
import jax.numpy as jnp
from jax._src import dtypes
from jax._src import test_util as jtu
import jax.scipy.signal as jsp_signal

jax.config.parse_flags_with_absl()

onedim_shapes = [(1,), (2,), (5,), (10,)]
twodim_shapes = [(1, 1), (2, 2), (2, 3), (3, 4), (4, 4)]
threedim_shapes = [(2, 2, 2), (3, 3, 2), (4, 4, 2), (5, 5, 2)]
stft_test_shapes = [
    # (input_shape, nperseg, noverlap, axis)
    ((50,), 17, 5, -1),
    ((2, 13), 7, 0, -1),
    ((3, 17, 2), 9, 3, 1),
    ((2, 3, 389, 5), 17, 13, 2),
    ((2, 1, 133, 3), 17, 13, -2),
    ((3, 7), 1, 0, 1),
]
csd_test_shapes = [
    # (x_input_shape, y_input_shape, nperseg, noverlap, axis)
    ((50,), (13,), 17, 5, -1),
    ((2, 13), (2, 13), 7, 0, -1),
    ((3, 17, 2), (3, 12, 2), 9, 3, 1),
]
welch_test_shapes = stft_test_shapes
istft_test_shapes = [
    # (input_shape, nperseg, noverlap, timeaxis, freqaxis)
    ((3, 2, 64, 31), 100, 75, -1, -2),
    ((17, 8, 5), 13, 7, 0, 1),
    ((65, 24), 24, 7, -2, -1),
]


default_dtypes = jtu.dtypes.floating + jtu.dtypes.integer + jtu.dtypes.complex
_TPU_FFT_TOL = 0.15

def _real_dtype(dtype):
  return jnp.finfo(dtypes.to_inexact_dtype(dtype)).dtype

def _complex_dtype(dtype):
  return dtypes.to_complex_dtype(dtype)


class LaxBackedScipySignalTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.stats implementations"""

  @jtu.sample_product(
    [dict(xshape=xshape, yshape=yshape)
     for shapeset in [onedim_shapes, twodim_shapes, threedim_shapes]
     for xshape in shapeset
     for yshape in shapeset
    ],
    mode=['full', 'same', 'valid'],
    op=['convolve', 'correlate'],
    method=['auto', 'direct', 'fft'],
    dtype=default_dtypes,
  )
  def testConvolutions(self, xshape, yshape, dtype, mode, op, method):
    jsp_op = getattr(jsp_signal, op)
    osp_op = getattr(osp_signal, op)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    osp_fun = partial(osp_op, mode=mode, method=method)
    jsp_fun = partial(jsp_op, mode=mode, method=method, precision=lax.Precision.HIGHEST)
    if method == 'fft':
      tol = {np.float16: 1e-2, np.float32: 1e-2, np.float64: 1e-6,
             np.complex64: 1e-2, np.complex128: 1e-6}
    else:
      tol = {np.float16: 1e-2, np.float32: 1e-2, np.float64: 1e-12,
             np.complex64: 1e-2, np.complex128: 1e-12}
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(
    [dict(xshape=xshape, yshape=yshape)
     for shapeset in [onedim_shapes, twodim_shapes, threedim_shapes]
     for xshape in shapeset
     for yshape in shapeset
    ],
    mode=['full', 'same', 'valid'],
    pass_axes=[True, False],
    dtype=default_dtypes,
  )
  def testFFTConvolution(self, xshape, yshape, dtype, mode, pass_axes):
    if pass_axes:
      # unspecified axes effectively act as batch dimensions, so their shape
      # must be equal
      axes = tuple(i for i in range(len(xshape)) if xshape[i] != yshape[i]) or 0
    else:
      axes = None
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    osp_fun = partial(osp_signal.fftconvolve, mode=mode, axes=axes)
    jsp_fun = partial(jsp_signal.fftconvolve, mode=mode, axes=axes)
    tol = {np.float16: 1e-2, np.float32: 1e-2, np.float64: 1e-6,
           np.complex64: 1e-2, np.complex128: 1e-6}
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, tol=tol)

  @jtu.sample_product(
    mode=['full', 'same', 'valid'],
    op=['convolve2d', 'correlate2d'],
    dtype=default_dtypes,
    xshape=twodim_shapes,
    yshape=twodim_shapes,
  )
  def testConvolutions2D(self, xshape, yshape, dtype, mode, op):
    jsp_op = getattr(jsp_signal, op)
    osp_op = getattr(osp_signal, op)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    osp_fun = partial(osp_op, mode=mode)
    jsp_fun = partial(jsp_op, mode=mode, precision=lax.Precision.HIGHEST)
    tol = {np.float16: 1e-2, np.float32: 1e-2, np.float64: 1e-12, np.complex64: 1e-2, np.complex128: 1e-12}
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(
    shape=[(5,), (4, 5), (3, 4, 5)],
    dtype=jtu.dtypes.floating + jtu.dtypes.integer,
    axis=[0, -1],
    type=['constant', 'linear'],
    bp=[0, [0, 2]],
  )
  def testDetrend(self, shape, dtype, axis, type, bp):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    kwds = dict(axis=axis, type=type, bp=bp)

    def osp_fun(x):
      return osp_signal.detrend(x, **kwds).astype(dtypes.to_inexact_dtype(x.dtype))
    jsp_fun = partial(jsp_signal.detrend, **kwds)

    tol = {np.float32: 1e-5, np.float64: 1e-12}

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, tol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(
    [dict(shape=shape, nperseg=nperseg, noverlap=noverlap, timeaxis=timeaxis,
          nfft=nfft)
      for shape, nperseg, noverlap, timeaxis in stft_test_shapes
      for nfft in [None, nperseg, int(nperseg * 1.5), nperseg * 2]
    ],
    dtype=default_dtypes,
    fs=[1.0, 16000.0],
    window=['boxcar', 'triang', 'blackman', 'hamming', 'hann'],
    detrend=['constant', 'linear', False],
    boundary=[None, 'even', 'odd', 'zeros'],
    padded=[True, False],
  )
  def testStftAgainstNumpy(self, *, shape, dtype, fs, window, nperseg,
                           noverlap, nfft, detrend, boundary, padded,
                           timeaxis):
    is_complex = dtypes.issubdtype(dtype, np.complexfloating)
    if is_complex and detrend is not None:
      self.skipTest("Complex signal is not supported in lax-backed `signal.detrend`.")

    kwds = dict(fs=fs, window=window, nfft=nfft, boundary=boundary, padded=padded,
                detrend=detrend, nperseg=nperseg, noverlap=noverlap, axis=timeaxis,
                return_onesided=not is_complex)

    def osp_fun(x):
      freqs, time, Pxx = osp_signal.stft(x, **kwds)
      return freqs.astype(_real_dtype(dtype)), time.astype(_real_dtype(dtype)), Pxx.astype(_complex_dtype(dtype))
    jsp_fun = partial(jsp_signal.stft, **kwds)
    tol = {
        np.float32: 1e-5, np.float64: 1e-12,
        np.complex64: 1e-5, np.complex128: 1e-12
    }
    if jtu.test_device_matches(['tpu']):
      tol = _TPU_FFT_TOL

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, rtol=tol, atol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

  # Tests with `average == 'median'`` is excluded from `testCsd*`
  # due to the issue:
  #   https://github.com/scipy/scipy/issues/15601
  @jtu.sample_product(
    [dict(xshape=xshape, yshape=yshape, nperseg=nperseg, noverlap=noverlap,
          timeaxis=timeaxis, nfft=nfft)
      for xshape, yshape, nperseg, noverlap, timeaxis in csd_test_shapes
      for nfft in [None, nperseg, int(nperseg * 1.5), nperseg * 2]
    ],
    dtype=default_dtypes,
    fs=[1.0, 16000.0],
    window=['boxcar', 'triang', 'blackman', 'hamming', 'hann'],
    detrend=['constant', 'linear', False],
    scaling=['density', 'spectrum'],
    average=['mean'],
  )
  def testCsdAgainstNumpy(
      self, *, xshape, yshape, dtype, fs, window, nperseg, noverlap, nfft,
      detrend, scaling, timeaxis, average):
    is_complex = dtypes.issubdtype(dtype, np.complexfloating)
    if is_complex and detrend is not None:
      self.skipTest("Complex signal is not supported in lax-backed `signal.detrend`.")

    kwds = dict(fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                nfft=nfft, detrend=detrend, return_onesided=not is_complex,
                scaling=scaling, axis=timeaxis, average=average)

    def osp_fun(x, y):
      freqs, Pxy = osp_signal.csd(x, y, **kwds)
      # Make type-casting the same as JAX.
      return freqs.astype(_real_dtype(dtype)), Pxy.astype(_complex_dtype(dtype))
    jsp_fun = partial(jsp_signal.csd, **kwds)
    tol = {
        np.float32: 1e-5, np.float64: 1e-12,
        np.complex64: 1e-5, np.complex128: 1e-12
    }
    if jtu.test_device_matches(['tpu']):
      tol = _TPU_FFT_TOL

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, rtol=tol, atol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(
    [dict(shape=shape, nperseg=nperseg, noverlap=noverlap, timeaxis=timeaxis,
          nfft=nfft)
      for shape, _yshape, nperseg, noverlap, timeaxis in csd_test_shapes
      for nfft in [None, nperseg, int(nperseg * 1.5), nperseg * 2]
    ],
    dtype=default_dtypes,
    fs=[1.0, 16000.0],
    window=['boxcar', 'triang', 'blackman', 'hamming', 'hann'],
    detrend=['constant', 'linear', False],
    scaling=['density', 'spectrum'],
    average=['mean'],
  )
  def testCsdWithSameParamAgainstNumpy(
      self, *, shape, dtype, fs, window, nperseg, noverlap, nfft,
      detrend, scaling, timeaxis, average):
    is_complex = dtypes.issubdtype(dtype, np.complexfloating)
    if is_complex and detrend is not None:
      self.skipTest("Complex signal is not supported in lax-backed `signal.detrend`.")

    kwds = dict(fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                nfft=nfft, detrend=detrend, return_onesided=not is_complex,
                scaling=scaling, axis=timeaxis, average=average)

    def osp_fun(x, y):
      # When the identical parameters are given, jsp-version follows
      # the behavior with copied parameters.
      freqs, Pxy = osp_signal.csd(x, y.copy(), **kwds)
      # Make type-casting the same as JAX.
      return freqs.astype(_real_dtype(dtype)), Pxy.astype(_complex_dtype(dtype))
    jsp_fun = partial(jsp_signal.csd, **kwds)
    tol = {
        np.float32: 1e-5, np.float64: 1e-12,
        np.complex64: 1e-5, np.complex128: 1e-12
    }
    if jtu.test_device_matches(['tpu']):
      tol = _TPU_FFT_TOL

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)] * 2

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, rtol=tol, atol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(
    [dict(shape=shape, nperseg=nperseg, noverlap=noverlap, timeaxis=timeaxis,
          nfft=nfft)
      for shape, nperseg, noverlap, timeaxis in welch_test_shapes
      for nfft in [None, nperseg, int(nperseg * 1.5), nperseg * 2]
    ],
    dtype=default_dtypes,
    fs=[1.0, 16000.0],
    window=['boxcar', 'triang', 'blackman', 'hamming', 'hann'],
    detrend=['constant', 'linear', False],
    return_onesided=[True, False],
    scaling=['density', 'spectrum'],
    average=['mean', 'median'],
  )
  def testWelchAgainstNumpy(self, *, shape, dtype, fs, window, nperseg,
                            noverlap, nfft, detrend, return_onesided,
                            scaling, timeaxis, average):
    if np.dtype(dtype).kind == 'c':
      return_onesided = False
      if detrend is not None:
        raise unittest.SkipTest(
            "Complex signal is not supported in lax-backed `signal.detrend`.")

    kwds = dict(fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                detrend=detrend, return_onesided=return_onesided, scaling=scaling,
                axis=timeaxis, average=average)

    def osp_fun(x):
      freqs, Pxx = osp_signal.welch(x, **kwds)
      return freqs.astype(_real_dtype(dtype)), Pxx.astype(_real_dtype(dtype))
    jsp_fun = partial(jsp_signal.welch, **kwds)
    tol = {
        np.float32: 1e-5, np.float64: 1e-12,
        np.complex64: 1e-5, np.complex128: 1e-12
    }
    if jtu.test_device_matches(['tpu']):
      tol = _TPU_FFT_TOL

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, rtol=tol, atol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(
    [dict(shape=shape, nperseg=nperseg, noverlap=noverlap, timeaxis=timeaxis)
      for shape, nperseg, noverlap, timeaxis in welch_test_shapes
    ],
    use_nperseg=[False, True],
    use_window=[False, True],
    use_noverlap=[False, True],
    dtype=jtu.dtypes.floating + jtu.dtypes.integer,
  )
  def testWelchWithDefaultStepArgsAgainstNumpy(
      self, *, shape, dtype, nperseg, noverlap, use_nperseg, use_noverlap,
      use_window, timeaxis):
    if tuple(shape) == (2, 3, 389, 5) and nperseg == 17 and noverlap == 13:
      raise unittest.SkipTest("Test fails for these inputs")
    kwargs = {'axis': timeaxis}

    if use_nperseg:
      kwargs['nperseg'] = nperseg
    if use_window:
      kwargs['window'] = jnp.array(osp_signal.get_window('hann', nperseg),
                                   dtype=dtypes.to_complex_dtype(dtype))
    if use_noverlap:
      kwargs['noverlap'] = noverlap

    @jtu.ignore_warning(message="nperseg = 256 is greater than")
    def osp_fun(x):
      freqs, Pxx = osp_signal.welch(x, **kwargs)
      return freqs.astype(_real_dtype(dtype)), Pxx.astype(_real_dtype(dtype))
    jsp_fun = partial(jsp_signal.welch, **kwargs)
    tol = {
        np.float32: 1e-5, np.float64: 1e-12,
        np.complex64: 1e-5, np.complex128: 1e-12
    }
    if jtu.test_device_matches(['tpu']):
      tol = _TPU_FFT_TOL

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, rtol=tol, atol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(
    [dict(shape=shape, nperseg=nperseg, noverlap=noverlap, timeaxis=timeaxis,
          freqaxis=freqaxis, nfft=nfft)
      for shape, nperseg, noverlap, timeaxis, freqaxis in istft_test_shapes
      for nfft in [None, nperseg, int(nperseg * 1.5), nperseg * 2]
    ],
    dtype=default_dtypes,
    fs=[1.0, 16000.0],
    window=['boxcar', 'triang', 'blackman', 'hamming', 'hann'],
    onesided=[False, True],
    boundary=[False, True],
  )
  def testIstftAgainstNumpy(self, *, shape, dtype, fs, window, nperseg,
                            noverlap, nfft, onesided, boundary,
                            timeaxis, freqaxis):
    if not onesided:
      new_freq_len = (shape[freqaxis] - 1) * 2
      shape = shape[:freqaxis] + (new_freq_len ,) + shape[freqaxis + 1:]

    kwds = dict(fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                nfft=nfft, input_onesided=onesided, boundary=boundary,
                time_axis=timeaxis, freq_axis=freqaxis)

    osp_fun = partial(osp_signal.istft, **kwds)
    osp_fun = jtu.ignore_warning(message="NOLA condition failed, STFT may not be invertible")(osp_fun)
    jsp_fun = partial(jsp_signal.istft, **kwds)

    tol = {
        np.float32: 1e-4, np.float64: 1e-6,
        np.complex64: 1e-4, np.complex128: 1e-6
    }
    if jtu.test_device_matches(['tpu']):
      tol = _TPU_FFT_TOL

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    # Here, dtype of output signal is different depending on osp versions,
    # and so depending on the test environment.  Thus, dtype check is disabled.
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, rtol=tol, atol=tol,
                            check_dtypes=False)
    self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
