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

from typing import Sequence, Union

import jax
from jax import lax


def unfold(
    input: jax.Array,
    kernel_size: Union[int, Sequence[int]],
    dilation: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    stride: Union[int, Sequence[int]] = 1
) -> jax.Array:
  """Extracts sliding local blocks from a batched `input` tensor.

  This is a wrapper around `lax.conv_general_dilated_patches`
  following the API of `torch.nn.Unfold`.

  See Also:
    https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

  Args:
    input: batch of images of shape `(N, C, H, W, ...)`.
    kernel_size: the size of the sliding blocks.
    dilation: a parameter that controls the stride of elements within the
      neighborhood.
    padding: implicit zero padding to be added on both sides of input.
    stride: the stride of the sliding blocks in the input spatial dimensions.

  Returns:
    A 3D tensor of shape `(N, C * np.prod(kernel_size), L)`, where `L` is the
    total number of patches.
  """
  ndim = input.ndim

  def to_tuple(x: Union[int, Sequence[int]]) -> Sequence[int]:
    if isinstance(x, int):
      x = (x,) * (ndim - 2)
    return x

  lhs_spec = rhs_spec = out_spec = tuple(range(ndim))
  patches = lax.conv_general_dilated_patches(
    lhs=input,
    filter_shape=to_tuple(kernel_size),
    window_strides=to_tuple(stride),
    padding=tuple((p, p) for p in to_tuple(padding)),
    rhs_dilation=to_tuple(dilation),
    dimension_numbers=lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)
  )  # NCHW...
  return patches.reshape(patches.shape[:2] + (-1,))  # NCL
