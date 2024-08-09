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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570

from jax._src.scipy.special import (
  bernoulli as bernoulli,
  bessel_jn as bessel_jn,
  beta as beta,
  betainc as betainc,
  betaln as betaln,
  digamma as digamma,
  entr as entr,
  erf as erf,
  erfc as erfc,
  erfinv as erfinv,
  exp1 as exp1,
  expi as expi,
  expit as expit,
  expn as expn,
  factorial as factorial,
  gamma as gamma,
  gammainc as gammainc,
  gammaincc as gammaincc,
  gammaln as gammaln,
  gammasgn as gammasgn,
  hyp1f1 as hyp1f1,
  i0 as i0,
  i0e as i0e,
  i1 as i1,
  i1e as i1e,
  kl_div as kl_div,
  log_ndtr as log_ndtr,
  log_softmax as log_softmax,
  logit as logit,
  logsumexp as logsumexp,
  lpmn as lpmn,
  lpmn_values as lpmn_values,
  multigammaln as multigammaln,
  ndtr as ndtr,
  ndtri as ndtri,
  poch as poch,
  polygamma as polygamma,
  rel_entr as rel_entr,
  softmax as softmax,
  spence as spence,
  sph_harm as sph_harm,
  xlog1py as xlog1py,
  xlogy as xlogy,
  zeta as zeta,
)

from jax._src.third_party.scipy.special import (
  fresnel as fresnel,
)
