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

from jax.numpy.linalg import (
  cholesky as cholesky,
  cross as cross,
  det as det,
  diagonal as diagonal,
  eigh as eigh,
  eigvalsh as eigvalsh,
  inv as inv,
  matmul as matmul,
  matrix_norm as matrix_norm,
  matrix_power as matrix_power,
  matrix_transpose as matrix_transpose,
  outer as outer,
  qr as qr,
  slogdet as slogdet,
  solve as solve,
  svd as svd,
  svdvals as svdvals,
  tensordot as tensordot,
  vecdot as vecdot,
  vector_norm as vector_norm,
)

from jax.numpy.linalg import trace as trace

from jax.experimental.array_api._linear_algebra_functions import (
  matrix_rank as matrix_rank,
  pinv as pinv,
)
