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

import jax.numpy as jnp

# TODO(micky774): Remove after adding device parameter to corresponding jnp.fft
# functions.
def fftfreq(n, /, *, d=1.0, device=None):
  """Returns the discrete Fourier transform sample frequencies."""
  return jnp.fft.fftfreq(n, d=d).to_device(device)

def rfftfreq(n, /, *, d=1.0, device=None):
  """Returns the discrete Fourier transform sample frequencies (for rfft and irfft)."""
  return jnp.fft.rfftfreq(n, d=d).to_device(device)
