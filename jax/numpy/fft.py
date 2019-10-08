# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from .. import lax
from ..lib.xla_bridge import xla_client, canonicalize_dtype
from ..util import get_module_functions
from .lax_numpy import _not_implemented
from .lax_numpy import _wraps
from . import lax_numpy as np
def _promote_to_complex(arg):
  dtype = np.result_type(arg, onp.complex64)
  # XLA's FFT op only supports C64.
  if dtype == onp.complex128:
    dtype = onp.complex64
  return lax.convert_element_type(arg, dtype)
@_wraps(onp.fft.fftn)
def fftn(a, s=None, axes=None, norm=None):
  # TODO(skye): implement padding/cropping based on 's'.
  if s is not None:
    raise NotImplementedError("jax.np.fftn only supports s=None, got %s" % s)
  if norm is not None:
    raise NotImplementedError("jax.np.fftn only supports norm=None, got %s" % norm)
  if s is not None and axes is not None and len(s) != len(axes):
    # Same error as numpy.
    raise ValueError("Shape and axes have different lengths.")

  orig_axes = axes
  if axes is None:
    if s is None:
      axes = range(a.ndim)
    else:
      axes = range(a.ndim - len(s), a.ndim)

  # XLA doesn't support 0-rank axes.
  if len(axes) == 0:
    return a

  if len(axes) != len(set(axes)):
    raise ValueError("jax.np.fftn does not support repeated axes. Got axes %s." % axes)

  if any(axis in range(a.ndim - 3) for axis in axes):
    raise ValueError("jax.np.fftn only supports 1D, 2D, and 3D FFTs over the innermost axes."
                     " Got axes %s with input rank %s." % (orig_axes, a.ndim))

  if s is None:
    s = [a.shape[axis] for axis in axes]
  a = _promote_to_complex(a)
  return lax.fft(a, xla_client.FftType.FFT, s)
for func in get_module_functions(onp.fft):
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
