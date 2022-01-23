# Copyright 2019 Google LLC
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

"""
Common neural network layer initializers, consistent with definitions
used in Keras and Sonnet.
"""


from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import lax
from jax import random
from jax import core
from jax._src.util import prod
from jax import dtypes

def zeros(key, shape, dtype=jnp.float_): return jnp.zeros(shape, dtypes.canonicalize_dtype(dtype))
def ones(key, shape, dtype=jnp.float_): return jnp.ones(shape, dtypes.canonicalize_dtype(dtype))

def uniform(scale=1e-2, dtype=jnp.float_):
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.uniform(key, shape, dtype) * scale
  return init

def normal(stddev=1e-2, dtype=jnp.float_):
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.normal(key, shape, dtype) * stddev
  return init

def truncated_normal(stddev=1e-2, lower=-2, upper=2, dtype=jnp.float_):
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.truncated_normal(key, lower, upper, shape, dtype) * stddev
  return init

def _compute_fans(shape: core.NamedShape, in_axis=-2, out_axis=-1):
  if isinstance(in_axis, int):
    in_size = shape[in_axis]
  else:
    in_size = int(np.prod([shape[i] for i in in_axis]))
  if isinstance(out_axis, int):
    out_size = shape[out_axis]
  else:
    out_size = int(np.prod([shape[i] for i in out_axis]))
  receptive_field_size = shape.total / in_size / out_size
  fan_in = in_size * receptive_field_size
  fan_out = out_size * receptive_field_size
  return fan_in, fan_out

def _complex_uniform(key, shape, dtype):
  """
  Sample uniform random values within a disk on the complex plane,
  with zero mean and unit variance.
  """
  key_r, key_theta = random.split(key)
  dtype = np.array(0, dtype).real.dtype
  r = jnp.sqrt(2 * random.uniform(key_r, shape, dtype))
  theta = 2 * jnp.pi * random.uniform(key_theta, shape, dtype)
  return r * jnp.exp(1j * theta)

def _complex_truncated_normal(key, upper, shape, dtype):
  """
  Sample random values from a centered normal distribution on the complex plane,
  whose modulus is truncated to `upper`, and the variance before the truncation is one.
  """
  key_r, key_theta = random.split(key)
  dtype = np.array(0, dtype).real.dtype
  t = (1 - jnp.exp(jnp.array(-(upper ** 2), dtype))) * random.uniform(key_r, shape, dtype)
  r = jnp.sqrt(-jnp.log(1 - t))
  theta = 2 * jnp.pi * random.uniform(key_theta, shape, dtype)
  return r * jnp.exp(1j * theta)

def variance_scaling(scale, mode, distribution, in_axis=-2, out_axis=-1, dtype=jnp.float_):
  """
  Initializer capable of adapting its scale to the shape of the weights tensor.

  With `distribution="truncated_normal" or "normal"`, samples are
  drawn from a truncated/untruncated normal distribution with a mean of zero and
  a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`,
  where `n` is:
  - number of input units in the weights tensor, if `mode="fan_in"`
  - number of output units, if `mode="fan_out"`
  - average of the numbers of input and output units, if `mode="fan_avg"`

  With `distribution="truncated_normal"`, the absolute values of the samples are
  truncated below 2 standard deviations before truncation.

  With `distribution="uniform"`, samples are drawn from:
  - a uniform interval, if `dtype` is real
  - a uniform disk, if `dtype` is complex
  with a mean of zero and a standard deviation of `stddev`.

  Args:
    scale: scaling factor (positive float).
    mode: one of "fan_in", "fan_out", and "fan_avg".
    distribution: random distribution to use. One of "truncated_normal",
      "normal" and "uniform".
    in_axis: axis or sequence of axes of the input dimension in the weights tensor.
    out_axis: axis or sequence of axes of the output dimension in the weights tensor.
    dtype: the dtype of the weights.
  """

  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    shape = core.as_named_shape(shape)
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in": denominator = fan_in
    elif mode == "fan_out": denominator = fan_out
    elif mode == "fan_avg": denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = jnp.array(scale / denominator, dtype=dtype)

    if distribution == "truncated_normal":
      if jnp.issubdtype(dtype, jnp.floating):
        # constant is stddev of standard normal truncated to (-2, 2)
        stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
        return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
      else:
        # constant is stddev of complex standard normal truncated to 2
        stddev = jnp.sqrt(variance) / jnp.array(.95311164380491208, dtype)
        return _complex_truncated_normal(key, 2, shape, dtype) * stddev
    elif distribution == "normal":
      return random.normal(key, shape, dtype) * jnp.sqrt(variance)
    elif distribution == "uniform":
      if jnp.issubdtype(dtype, jnp.floating):
        return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)
      else:
        return _complex_uniform(key, shape, dtype) * jnp.sqrt(variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer: {}".format(distribution))

  return init

xavier_uniform = glorot_uniform = partial(variance_scaling, 1.0, "fan_avg", "uniform")
xavier_normal = glorot_normal = partial(variance_scaling, 1.0, "fan_avg", "truncated_normal")
lecun_uniform = partial(variance_scaling, 1.0, "fan_in", "uniform")
lecun_normal = partial(variance_scaling, 1.0, "fan_in", "truncated_normal")
kaiming_uniform = he_uniform = partial(variance_scaling, 2.0, "fan_in", "uniform")
kaiming_normal = he_normal = partial(variance_scaling, 2.0, "fan_in", "truncated_normal")

def orthogonal(scale=1.0, column_axis=-1, dtype=jnp.float_):
  """
  Construct an initializer for uniformly distributed orthogonal matrices.

  If the shape is not square, the matrices will have orthonormal rows or columns
  depending on which side is smaller.
  """
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    if len(shape) < 2:
      raise ValueError("orthogonal initializer requires at least a 2D shape")
    n_rows, n_cols = prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_cols, n_rows) if n_rows < n_cols else (n_rows, n_cols)
    A = random.normal(key, matrix_shape, dtype)
    Q, R = jnp.linalg.qr(A)
    diag_sign = lax.broadcast_to_rank(jnp.sign(jnp.diag(R)), rank=Q.ndim)
    Q *= diag_sign # needed for a uniform distribution
    if n_rows < n_cols: Q = Q.T
    Q = jnp.reshape(Q, tuple(np.delete(shape, column_axis)) + (shape[column_axis],))
    Q = jnp.moveaxis(Q, -1, column_axis)
    return scale * Q
  return init


def delta_orthogonal(scale=1.0, column_axis=-1, dtype=jnp.float_):
  """
  Construct an initializer for delta orthogonal kernels; see arXiv:1806.05393.

  The shape must be 3D, 4D or 5D.
  """
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Delta orthogonal initializer requires a 3D, 4D or 5D "
                       "shape.")
    if shape[-1] < shape[-2]:
      raise ValueError("`fan_in` must be less or equal than `fan_out`. ")
    ortho_init = orthogonal(scale=scale, column_axis=column_axis, dtype=dtype)
    ortho_matrix = ortho_init(key, shape[-2:])
    W = jnp.zeros(shape, dtype=dtype)
    if len(shape) == 3:
      k = shape[0]
      return W.at[(k-1)//2, ...].set(ortho_matrix)
    elif len(shape) == 4:
      k1, k2 = shape[:2]
      return W.at[(k1-1)//2, (k2-1)//2, ...].set(ortho_matrix)
    else:
      k1, k2, k3 = shape[:3]
      return W.at[(k1-1)//2, (k2-1)//2, (k3-1)//2, ...].set(ortho_matrix)
  return init
