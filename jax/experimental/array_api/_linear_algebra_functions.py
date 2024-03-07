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

import jax

# TODO(micky774): Remove after deprecation is completed (began 2024-5-14)
def matrix_rank(x, /, *, rtol=None):
  """
  Returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of matrices).
  """
  return jax.numpy.linalg.matrix_rank(x, rtol)

def pinv(x, /, *, rtol=None):
  """
  Returns the (Moore-Penrose) pseudo-inverse of a matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.pinv(x, rtol)
