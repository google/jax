# Copyright 2021 The JAX Authors.
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
# limitations under the License


import scipy.stats as osp_stats

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import implements, promote_args_inexact
from jax._src.scipy.special import betaln
from jax._src.typing import Array, ArrayLike


@implements(osp_stats.betabinom.logpmf, update_doc=False)
def logpmf(k: ArrayLike, n: ArrayLike, a: ArrayLike, b: ArrayLike,
           loc: ArrayLike = 0) -> Array:
  """JAX implementation of scipy.stats.betabinom.logpmf."""
  k, n, a, b, loc = promote_args_inexact("betabinom.logpmf", k, n, a, b, loc)
  y = lax.sub(lax.floor(k), loc)
  one = _lax_const(y, 1)
  zero = _lax_const(y, 0)
  combiln = lax.neg(lax.add(lax.log1p(n), betaln(lax.add(lax.sub(n,y), one), lax.add(y,one))))
  beta_lns = lax.sub(betaln(lax.add(y,a), lax.add(lax.sub(n,y),b)), betaln(a,b))
  log_probs = lax.add(combiln, beta_lns)
  y_cond = jnp.logical_or(lax.lt(y, lax.neg(loc)), lax.gt(y, lax.sub(n, loc)))
  log_probs = jnp.where(y_cond, -jnp.inf, log_probs)
  n_a_b_cond = jnp.logical_or(jnp.logical_or(lax.lt(n, zero), lax.lt(a, zero)), lax.lt(b, zero))
  return jnp.where(n_a_b_cond, jnp.nan, log_probs)


@implements(osp_stats.betabinom.pmf, update_doc=False)
def pmf(k: ArrayLike, n: ArrayLike, a: ArrayLike, b: ArrayLike,
        loc: ArrayLike = 0) -> Array:
  """JAX implementation of scipy.stats.betabinom.pmf."""
  return lax.exp(logpmf(k, n, a, b, loc))
