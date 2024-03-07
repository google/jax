# Copyright 2018 The JAX Authors.
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
import itertools
import unittest

from absl.testing import absltest

import numpy as np
import scipy.stats as osp_stats
import scipy.version

import jax
import jax.numpy as jnp
from jax._src import dtypes, test_util as jtu
from jax.scipy import stats as lsp_stats
from jax.scipy.special import expit

jax.config.parse_flags_with_absl()

scipy_version = jtu.parse_version(scipy.version.version)

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]
one_and_two_dim_shapes = [(4,), (3, 4), (3, 1), (1, 4)]


def genNamedParametersNArgs(n):
  return jtu.sample_product(
    shapes=itertools.combinations_with_replacement(all_shapes, n),
    dtypes=itertools.combinations_with_replacement(jtu.dtypes.floating, n),
  )


# Allow implicit rank promotion in these tests, as virtually every test exercises it.
@jtu.with_config(jax_numpy_rank_promotion="allow")
class LaxBackedScipyStatsTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.stats implementations"""

  @genNamedParametersNArgs(2)
  def testVonMisesPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.vonmises.pdf
    lax_fun = lsp_stats.vonmises.pdf

    def args_maker():
      x, kappa = map(rng, shapes, dtypes)
      kappa = np.where(kappa < 0, kappa * -1, kappa).astype(kappa.dtype)
      return [x, kappa]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(2)
  def testVonMisesLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.vonmises.pdf
    lax_fun = lsp_stats.vonmises.pdf

    def args_maker():
      x, kappa = map(rng, shapes, dtypes)
      kappa = np.where(kappa < 0, kappa * -1, kappa).astype(kappa.dtype)
      return [x, kappa]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(2)
  def testWrappedCauchyPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    rng_uniform = jtu.rand_uniform(self.rng(), low=1e-3, high=1 - 1e-3)
    scipy_fun = osp_stats.wrapcauchy.pdf
    lax_fun = lsp_stats.wrapcauchy.pdf

    def args_maker():
      x = rng(shapes[0], dtypes[0])
      c = rng_uniform(shapes[1], dtypes[1])
      return [x, c]

    tol = {
        np.float32: 1e-4 if jtu.test_device_matches(["tpu"]) else 1e-5,
        np.float64: 1e-11,
    }
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                              check_dtypes=False, tol=tol)
      self._CompileAndCheck(lax_fun, args_maker, tol=tol)

  @genNamedParametersNArgs(2)
  def testWrappedCauchyLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    rng_uniform = jtu.rand_uniform(self.rng(), low=1e-3, high=1 - 1e-3)
    scipy_fun = osp_stats.wrapcauchy.logpdf
    lax_fun = lsp_stats.wrapcauchy.logpdf

    def args_maker():
      x = rng(shapes[0], dtypes[0])
      c = rng_uniform(shapes[1], dtypes[1])
      return [x, c]

    tol = {
        np.float32: 1e-4 if jtu.test_device_matches(["tpu"]) else 1e-5,
        np.float64: 1e-11,
    }
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                              check_dtypes=False, tol=tol)
      self._CompileAndCheck(lax_fun, args_maker, tol=tol)

  @genNamedParametersNArgs(3)
  def testPoissonLogPmf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.poisson.logpmf
    lax_fun = lsp_stats.poisson.logpmf

    def args_maker():
      k, mu, loc = map(rng, shapes, dtypes)
      # clipping to ensure that rate parameter is strictly positive
      mu = np.clip(np.abs(mu), a_min=0.1, a_max=None).astype(mu.dtype)
      loc = np.floor(loc)
      return [k, mu, loc]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker, rtol={np.float64: 1e-14})

  @genNamedParametersNArgs(3)
  def testPoissonPmf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.poisson.pmf
    lax_fun = lsp_stats.poisson.pmf

    def args_maker():
      k, mu, loc = map(rng, shapes, dtypes)
      # clipping to ensure that rate parameter is strictly positive
      mu = np.clip(np.abs(mu), a_min=0.1, a_max=None).astype(mu.dtype)
      loc = np.floor(loc)
      return [k, mu, loc]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testPoissonCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.poisson.cdf
    lax_fun = lsp_stats.poisson.cdf

    def args_maker():
      k, mu, loc = map(rng, shapes, dtypes)
      # clipping to ensure that rate parameter is strictly positive
      mu = np.clip(np.abs(mu), a_min=0.1, a_max=None).astype(mu.dtype)
      return [k, mu, loc]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(3)
  def testBernoulliLogPmf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.bernoulli.logpmf
    lax_fun = lsp_stats.bernoulli.logpmf

    def args_maker():
      x, logit, loc = map(rng, shapes, dtypes)
      x = np.floor(x)
      p = expit(logit)
      loc = np.floor(loc)
      return [x, p, loc]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(2)
  def testBernoulliCdf(self, shapes, dtypes):
    rng_int = jtu.rand_int(self.rng(), -100, 100)
    rng_uniform = jtu.rand_uniform(self.rng())
    scipy_fun = osp_stats.bernoulli.cdf
    lax_fun = lsp_stats.bernoulli.cdf

    def args_maker():
      x = rng_int(shapes[0], dtypes[0])
      p = rng_uniform(shapes[1], dtypes[1])
      return [x, p]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(2)
  def testBernoulliPpf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.bernoulli.ppf
    lax_fun = lsp_stats.bernoulli.ppf

    if scipy_version < (1, 9, 2):
      self.skipTest("Scipy 1.9.2 needed for fix https://github.com/scipy/scipy/pull/17166.")

    def args_maker():
      q, p = map(rng, shapes, dtypes)
      q = expit(q)
      p = expit(p)
      return [q, p]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                             tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=3e-4)

  @genNamedParametersNArgs(3)
  def testGeomLogPmf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.geom.logpmf
    lax_fun = lsp_stats.geom.logpmf

    def args_maker():
      x, logit, loc = map(rng, shapes, dtypes)
      x = np.floor(x)
      p = expit(logit)
      loc = np.floor(loc)
      return [x, p, loc]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(5)
  def testBetaLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.beta.logpdf
    lax_fun = lsp_stats.beta.logpdf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker,
                            rtol={np.float32: 2e-3, np.float64: 1e-4})

  @genNamedParametersNArgs(5)
  def testBetaLogCdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.beta.logcdf
    lax_fun = lsp_stats.beta.logcdf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker,
                            rtol={np.float32: 2e-3, np.float64: 1e-4})

  @genNamedParametersNArgs(5)
  def testBetaSf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.beta.sf
    lax_fun = lsp_stats.beta.sf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker,
                            rtol={np.float32: 2e-3, np.float64: 1e-4})

  @genNamedParametersNArgs(5)
  def testBetaLogSf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.beta.logsf
    lax_fun = lsp_stats.beta.logsf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker,
                            rtol={np.float32: 2e-3, np.float64: 1e-4})

  def testBetaLogPdfZero(self):
    # Regression test for https://github.com/google/jax/issues/7645
    a = b = 1.
    x = np.array([0., 1.])
    self.assertAllClose(
      osp_stats.beta.pdf(x, a, b), lsp_stats.beta.pdf(x, a, b), atol=1e-5,
      rtol=2e-5)

  def testBetaLogPdfNegativeConstants(self):
    a = b = -1.1
    x = jnp.array([0., 0.5, 1.])
    self.assertAllClose(
      osp_stats.beta.pdf(x, a, b), lsp_stats.beta.pdf(x, a, b), atol=1e-5,
      rtol=2e-5)

  def testBetaLogPdfNegativeScale(self):
    a = b = 1.
    x = jnp.array([0., 0.5, 1.])
    loc = 0
    scale = -1
    self.assertAllClose(
      osp_stats.beta.pdf(x, a, b, loc, scale),
      lsp_stats.beta.pdf(x, a, b, loc, scale), atol=1e-5,
      rtol=2e-5)

  @genNamedParametersNArgs(3)
  def testCauchyLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.cauchy.logpdf
    lax_fun = lsp_stats.cauchy.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker, tol={np.float64: 1E-14})

  @genNamedParametersNArgs(3)
  def testCauchyLogCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.cauchy.logcdf
    lax_fun = lsp_stats.cauchy.logcdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol={np.float64: 1e-14},
                            atol={np.float64: 1e-14})

  @genNamedParametersNArgs(3)
  def testCauchyCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.cauchy.cdf
    lax_fun = lsp_stats.cauchy.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol={np.float64: 1e-14},
                            atol={np.float64: 1e-14})

  @genNamedParametersNArgs(3)
  def testCauchyLogSf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.cauchy.logsf
    lax_fun = lsp_stats.cauchy.logsf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol={np.float64: 1e-14},
                            atol={np.float64: 1e-14})

  @genNamedParametersNArgs(3)
  def testCauchySf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.cauchy.sf
    lax_fun = lsp_stats.cauchy.sf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol={np.float64: 1e-14},
                            atol={np.float64: 1e-14})

  @genNamedParametersNArgs(3)
  def testCauchyIsf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.cauchy.isf
    lax_fun = lsp_stats.cauchy.isf

    def args_maker():
      q, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that q is in desired range
      # since lax.tan and numpy.tan work different near divergence points
      q = np.clip(q, 5e-3, 1 - 5e-3).astype(q.dtype)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [q, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=2e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=3e-4)

  @genNamedParametersNArgs(3)
  def testCauchyPpf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.cauchy.ppf
    lax_fun = lsp_stats.cauchy.ppf

    def args_maker():
      q, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that q is in desired
      # since lax.tan and numpy.tan work different near divergence points
      q = np.clip(q, 5e-3, 1 - 5e-3).astype(q.dtype)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [q, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=2e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=3e-4)

  @jtu.sample_product(
    shapes=[
      [x_shape, alpha_shape]
      for x_shape in one_and_two_dim_shapes
      for alpha_shape in [(x_shape[0],), (x_shape[0] + 1,)]
    ],
    dtypes=itertools.combinations_with_replacement(jtu.dtypes.floating, 2),
  )
  def testDirichletLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())

    def _normalize(x, alpha):
      x_norm = x.sum(0) + (0.0 if x.shape[0] == alpha.shape[0] else 0.1)
      return (x / x_norm).astype(x.dtype), alpha

    def lax_fun(x, alpha):
      return lsp_stats.dirichlet.logpdf(*_normalize(x, alpha))

    def scipy_fun(x, alpha):
      # scipy validates the x normalization using float64 arithmetic, so we must
      # cast x to float64 before normalization to ensure this passes.
      x, alpha = _normalize(x.astype('float64'), alpha)

      result = osp_stats.dirichlet.logpdf(x, alpha)
      # if x.shape is (N, 1), scipy flattens the output, while JAX returns arrays
      # of a consistent rank. This check ensures the results have the same shape.
      return result if x.ndim == 1 else np.atleast_1d(result)

    def args_maker():
      # Don't normalize here, because we want normalization to happen at 64-bit
      # precision in the scipy version.
      x, alpha = map(rng, shapes, dtypes)
      return x, alpha

    tol = {np.float32: 1E-3, np.float64: 1e-5}

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=tol)
      self._CompileAndCheck(lax_fun, args_maker, atol=tol, rtol=tol)

  @genNamedParametersNArgs(3)
  def testExponLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.expon.logpdf
    lax_fun = lsp_stats.expon.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testGammaLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.gamma.logpdf
    lax_fun = lsp_stats.gamma.logpdf

    def args_maker():
      x, a, loc, scale = map(rng, shapes, dtypes)
      return [x, a, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  def testGammaLogPdfZero(self):
    # Regression test for https://github.com/google/jax/issues/7256
    self.assertAllClose(
      osp_stats.gamma.pdf(0.0, 1.0), lsp_stats.gamma.pdf(0.0, 1.0), atol=1E-6)

  @genNamedParametersNArgs(4)
  def testGammaLogCdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.gamma.logcdf
    lax_fun = lsp_stats.gamma.logcdf

    def args_maker():
      x, a, loc, scale = map(rng, shapes, dtypes)
      x = np.clip(x, 0, None).astype(x.dtype)
      return [x, a, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testGammaLogSf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.gamma.logsf
    lax_fun = lsp_stats.gamma.logsf

    def args_maker():
      x, a, loc, scale = map(rng, shapes, dtypes)
      return [x, a, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testGammaSf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.gamma.sf
    lax_fun = lsp_stats.gamma.sf

    def args_maker():
      x, a, loc, scale = map(rng, shapes, dtypes)
      return [x, a, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(2)
  def testGenNormLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.gennorm.logpdf
    lax_fun = lsp_stats.gennorm.logpdf

    def args_maker():
      x, p = map(rng, shapes, dtypes)
      return [x, p]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4, rtol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(2)
  def testGenNormCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.gennorm.cdf
    lax_fun = lsp_stats.gennorm.cdf

    def args_maker():
      x, p = map(rng, shapes, dtypes)
      return [x, p]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4, rtol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker, atol={np.float32: 3e-5},
                            rtol={np.float32: 3e-5})

  @genNamedParametersNArgs(4)
  def testNBinomLogPmf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.nbinom.logpmf
    lax_fun = lsp_stats.nbinom.logpmf

    def args_maker():
      k, n, logit, loc = map(rng, shapes, dtypes)
      k = np.floor(np.abs(k))
      n = np.ceil(np.abs(n))
      p = expit(logit)
      loc = np.floor(loc)
      return [k, n, p, loc]

    tol = {np.float32: 1e-6, np.float64: 1e-8}

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=tol, atol=tol)

  @genNamedParametersNArgs(3)
  def testLaplaceLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.laplace.logpdf
    lax_fun = lsp_stats.laplace.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testLaplaceCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.laplace.cdf
    lax_fun = lsp_stats.laplace.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol={np.float32: 1e-5, np.float64: 1e-6})
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testLogisticCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.cdf
    lax_fun = lsp_stats.logistic.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=3e-5)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testLogisticLogpdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.logpdf
    lax_fun = lsp_stats.logistic.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  def testLogisticLogpdfOverflow(self):
    # Regression test for https://github.com/google/jax/issues/10219
    self.assertAllClose(
      np.array([-100, -100], np.float32),
      lsp_stats.logistic.logpdf(np.array([-100, 100], np.float32)),
      check_dtypes=False)

  @genNamedParametersNArgs(3)
  def testLogisticPpf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.ppf
    lax_fun = lsp_stats.logistic.ppf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              atol=1e-3, rtol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker, rtol=3e-4)

  @genNamedParametersNArgs(3)
  def testLogisticSf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.sf
    lax_fun = lsp_stats.logistic.sf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=2e-5)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testLogisticIsf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.isf
    lax_fun = lsp_stats.logistic.isf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=3e-4)

  @genNamedParametersNArgs(3)
  def testNormLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.logpdf
    lax_fun = lsp_stats.norm.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(3)
  def testNormLogCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.logcdf
    lax_fun = lsp_stats.norm.logcdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(3)
  def testNormCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.cdf
    lax_fun = lsp_stats.norm.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-6)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testNormLogSf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.logsf
    lax_fun = lsp_stats.norm.logsf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testNormSf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.sf
    lax_fun = lsp_stats.norm.sf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-6)
      self._CompileAndCheck(lax_fun, args_maker)

  def testNormSfNearZero(self):
    # Regression test for https://github.com/google/jax/issues/17199
    value = np.array(10, np.float32)
    self.assertAllClose(osp_stats.norm.sf(value).astype('float32'),
                        lsp_stats.norm.sf(value),
                        atol=0, rtol=1E-5)

  @genNamedParametersNArgs(3)
  def testNormPpf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.ppf
    lax_fun = lsp_stats.norm.ppf

    def args_maker():
      q, loc, scale = map(rng, shapes, dtypes)
      # ensure probability is between 0 and 1:
      q = np.clip(np.abs(q / 3), a_min=None, a_max=1).astype(q.dtype)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [q, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=3e-4)

  @genNamedParametersNArgs(3)
  def testNormIsf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.isf
    lax_fun = lsp_stats.norm.isf

    def args_maker():
      q, loc, scale = map(rng, shapes, dtypes)
      # ensure probability is between 0 and 1:
      q = np.clip(np.abs(q / 3), a_min=None, a_max=1).astype(q.dtype)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [q, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=3e-4, atol=3e-4)

  @genNamedParametersNArgs(5)
  def testTruncnormLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.truncnorm.logpdf
    lax_fun = lsp_stats.truncnorm.logpdf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)

      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(5)
  def testTruncnormPdf(self, shapes, dtypes):
    if jtu.test_device_matches(["cpu"]):
      raise unittest.SkipTest("TODO(b/282695039): test fails at LLVM head")
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.truncnorm.pdf
    lax_fun = lsp_stats.truncnorm.pdf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)

      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(5)
  def testTruncnormLogCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.truncnorm.logcdf
    lax_fun = lsp_stats.truncnorm.logcdf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)

      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(5)
  def testTruncnormCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.truncnorm.cdf
    lax_fun = lsp_stats.truncnorm.cdf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)

      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker, rtol={np.float32: 1e-5},
                            atol={np.float32: 1e-5})

  @genNamedParametersNArgs(5)
  def testTruncnormLogSf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.truncnorm.logsf
    lax_fun = lsp_stats.truncnorm.logsf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)

      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(5)
  def testTruncnormSf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.truncnorm.sf
    lax_fun = lsp_stats.truncnorm.sf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)

      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, a, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testParetoLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.pareto.logpdf
    lax_fun = lsp_stats.pareto.logpdf

    def args_maker():
      x, b, loc, scale = map(rng, shapes, dtypes)
      return [x, b, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(4)
  def testTLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.t.logpdf
    lax_fun = lsp_stats.t.logpdf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
      return [x, df, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-3)
      self._CompileAndCheck(lax_fun, args_maker,
                            rtol={np.float64: 1e-14}, atol={np.float64: 1e-14})


  @genNamedParametersNArgs(3)
  def testUniformLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.uniform.logpdf
    lax_fun = lsp_stats.uniform.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      return [x, loc, np.abs(scale)]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testUniformCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.uniform.cdf
    lax_fun = lsp_stats.uniform.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      return [x, loc, np.abs(scale)]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-5)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testUniformPpf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.uniform.ppf
    lax_fun = lsp_stats.uniform.ppf

    def args_maker():
      q, loc, scale = map(rng, shapes, dtypes)
      return [q, loc, np.abs(scale)]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=1e-5)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testChi2LogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.chi2.logpdf
    lax_fun = lsp_stats.chi2.logpdf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      return [x, df, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(4)
  def testChi2LogCdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.chi2.logcdf
    lax_fun = lsp_stats.chi2.logcdf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      return [x, df, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testChi2Cdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.chi2.cdf
    lax_fun = lsp_stats.chi2.cdf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      return [x, df, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testChi2Sf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.chi2.sf
    lax_fun = lsp_stats.chi2.sf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      return [x, df, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testChi2LogSf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.chi2.logsf
    lax_fun = lsp_stats.chi2.logsf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      return [x, df, loc, scale]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(5)
  def testBetaBinomLogPmf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    lax_fun = lsp_stats.betabinom.logpmf

    def args_maker():
      k, n, a, b, loc = map(rng, shapes, dtypes)
      k = np.floor(k)
      n = np.ceil(n)
      a = np.clip(a, a_min = 0.1, a_max=None).astype(a.dtype)
      b = np.clip(a, a_min = 0.1, a_max=None).astype(b.dtype)
      loc = np.floor(loc)
      return [k, n, a, b, loc]

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      scipy_fun = osp_stats.betabinom.logpmf
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=1e-5, atol=1e-5)

  @genNamedParametersNArgs(4)
  def testBinomLogPmf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.binom.logpmf
    lax_fun = lsp_stats.binom.logpmf

    def args_maker():
      k, n, logit, loc = map(rng, shapes, dtypes)
      k = np.floor(np.abs(k))
      n = np.ceil(np.abs(n))
      p = expit(logit)
      loc = np.floor(loc)
      return [k, n, p, loc]

    tol = {np.float32: 1e-6, np.float64: 1e-8}

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                              tol=5e-4)
      self._CompileAndCheck(lax_fun, args_maker, rtol=tol, atol=tol)

  def testBinomPmfOutOfRange(self):
    # Regression test for https://github.com/google/jax/issues/19150
    self.assertEqual(lsp_stats.binom.pmf(k=6.5, n=5, p=0.8), 0.0)

  def testIssue972(self):
    self.assertAllClose(
      np.ones((4,), np.float32),
      lsp_stats.norm.cdf(np.full((4,), np.inf, np.float32)),
      check_dtypes=False)

  @jtu.sample_product(
    [dict(x_dtype=x_dtype, p_dtype=p_dtype)
     for x_dtype, p_dtype in itertools.product(jtu.dtypes.integer, jtu.dtypes.floating)
    ],
    shape=[(2), (4,), (1, 5)],
  )
  def testMultinomialLogPmf(self, shape, x_dtype, p_dtype):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.multinomial.logpmf
    lax_fun = lsp_stats.multinomial.logpmf

    def args_maker():
      x = rng(shape, x_dtype)
      n = np.sum(x, dtype=x.dtype)
      p = rng(shape, p_dtype)
      # Normalize the array such that it sums it's entries sum to 1 (or close enough to)
      p = p / np.sum(p)
      return [x, n, p]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=5e-4)
    self._CompileAndCheck(lax_fun, args_maker, rtol=1e-5, atol=1e-5)

  @jtu.sample_product(
    [dict(x_shape=x_shape, mean_shape=mean_shape, cov_shape=cov_shape)
      for x_shape, mean_shape, cov_shape in [
          # # These test cases cover default values for mean/cov, but we don't
          # # support those yet (and they seem not very valuable).
          # [(), None, None],
          # [(), (), None],
          # [(2,), None, None],
          # [(2,), (), None],
          # [(2,), (2,), None],
          # [(3, 2), (3, 2,), None],
          # [(5, 3, 2), (5, 3, 2,), None],

          [(), (), ()],
          [(3,), (), ()],
          [(3,), (3,), ()],
          [(3,), (3,), (3, 3)],
          [(3, 4), (4,), (4, 4)],
          [(2, 3, 4), (4,), (4, 4)],
      ]
    ],
    [dict(x_dtype=x_dtype, mean_dtype=mean_dtype, cov_dtype=cov_dtype)
     for x_dtype, mean_dtype, cov_dtype in itertools.combinations_with_replacement(jtu.dtypes.floating, 3)
    ],
    # if (mean_shape is not None or mean_dtype == np.float32)
    # and (cov_shape is not None or cov_dtype == np.float32)))
  )
  def testMultivariateNormalLogpdf(self, x_shape, x_dtype, mean_shape,
                                   mean_dtype, cov_shape, cov_dtype):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      args = [rng(x_shape, x_dtype)]
      if mean_shape is not None:
        args.append(5 * rng(mean_shape, mean_dtype))
      if cov_shape is not None:
        if cov_shape == ():
          args.append(0.1 + rng(cov_shape, cov_dtype) ** 2)
        else:
          factor_shape = (*cov_shape[:-1], 2 * cov_shape[-1])
          factor = rng(factor_shape, cov_dtype)
          args.append(np.matmul(factor, np.swapaxes(factor, -1, -2)))
      return [a.astype(x_dtype) for a in args]

    self._CheckAgainstNumpy(osp_stats.multivariate_normal.logpdf,
                            lsp_stats.multivariate_normal.logpdf,
                            args_maker, tol=1e-3, check_dtypes=False)
    self._CompileAndCheck(lsp_stats.multivariate_normal.logpdf, args_maker,
                          rtol=1e-4, atol=1e-4)


  @jtu.sample_product(
    [dict(x_shape=x_shape, mean_shape=mean_shape, cov_shape=cov_shape)
      for x_shape, mean_shape, cov_shape in [
          # These test cases are where scipy flattens things, which has
          # different batch semantics than some might expect, so we manually
          # vectorize scipy's outputs for the sake of testing.
          [(5, 3, 2), (5, 3, 2), (5, 3, 2, 2)],
          [(2,), (5, 3, 2), (5, 3, 2, 2)],
          [(5, 3, 2), (2,), (5, 3, 2, 2)],
          [(5, 3, 2), (5, 3, 2,), (2, 2)],
          [(1, 3, 2), (3, 2,), (5, 1, 2, 2)],
          [(5, 3, 2), (1, 2,), (2, 2)],
      ]
    ],
    [dict(x_dtype=x_dtype, mean_dtype=mean_dtype, cov_dtype=cov_dtype)
     for x_dtype, mean_dtype, cov_dtype in itertools.combinations_with_replacement(jtu.dtypes.floating, 3)
    ],
  )
  def testMultivariateNormalLogpdfBroadcasted(self, x_shape, x_dtype, mean_shape,
                                              mean_dtype, cov_shape, cov_dtype):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      args = [rng(x_shape, x_dtype)]
      if mean_shape is not None:
        args.append(5 * rng(mean_shape, mean_dtype))
      if cov_shape is not None:
        if cov_shape == ():
          args.append(0.1 + rng(cov_shape, cov_dtype) ** 2)
        else:
          factor_shape = (*cov_shape[:-1], 2 * cov_shape[-1])
          factor = rng(factor_shape, cov_dtype)
          args.append(np.matmul(factor, np.swapaxes(factor, -1, -2)))
      return [a.astype(x_dtype) for a in args]

    osp_fun = np.vectorize(osp_stats.multivariate_normal.logpdf,
                           signature="(n),(n),(n,n)->()")

    self._CheckAgainstNumpy(osp_fun, lsp_stats.multivariate_normal.logpdf,
                            args_maker, tol=1e-3, check_dtypes=False)
    self._CompileAndCheck(lsp_stats.multivariate_normal.logpdf, args_maker,
                          rtol=1e-4, atol=1e-4)


  @jtu.sample_product(
    ndim=[2, 3],
    nbatch=[1, 3, 5],
    dtype=jtu.dtypes.floating,
  )
  def testMultivariateNormalLogpdfBatch(self, ndim, nbatch, dtype):
    # Regression test for #5570
    rng = jtu.rand_default(self.rng())
    x = rng((nbatch, ndim), dtype)
    mean = 5 * rng((nbatch, ndim), dtype)
    factor = rng((nbatch, ndim, 2 * ndim), dtype)
    cov = factor @ factor.transpose(0, 2, 1)

    result1 = lsp_stats.multivariate_normal.logpdf(x, mean, cov)
    result2 = jax.vmap(lsp_stats.multivariate_normal.logpdf)(x, mean, cov)
    self.assertArraysAllClose(result1, result2, check_dtypes=False)

  @jtu.sample_product(
    inshape=[(50,), (3, 50), (2, 12)],
    dtype=jtu.dtypes.floating,
    outsize=[None, 10],
    weights=[False, True],
    method=[None, "scott", "silverman", 1.5, "callable"],
    func=[None, "evaluate", "logpdf", "pdf"],
  )
  @jax.default_matmul_precision("float32")
  def testKde(self, inshape, dtype, outsize, weights, method, func):
    if method == "callable":
      method = lambda kde: kde.neff ** -1./(kde.d+4)

    def scipy_fun(dataset, points, w):
      w = np.abs(w) if weights else None
      kde = osp_stats.gaussian_kde(dataset, bw_method=method, weights=w)
      if func is None:
        result = kde(points)
      else:
        result = getattr(kde, func)(points)
      # Note: the scipy implementation _always_ returns float64
      return result.astype(dtype)

    def lax_fun(dataset, points, w):
      w = jax.numpy.abs(w) if weights else None
      kde = lsp_stats.gaussian_kde(dataset, bw_method=method, weights=w)
      if func is None:
        result = kde(points)
      else:
        result = getattr(kde, func)(points)
      return result

    if outsize is None:
      outshape = inshape
    else:
      outshape = inshape[:-1] + (outsize,)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [
      rng(inshape, dtype), rng(outshape, dtype), rng(inshape[-1:], dtype)]
    self._CheckAgainstNumpy(
        scipy_fun, lax_fun, args_maker, tol={
            np.float32: 2e-2 if jtu.test_device_matches(["tpu"]) else 1e-3,
            np.float64: 3e-14
        })
    self._CompileAndCheck(
        lax_fun, args_maker, rtol={np.float32: 3e-5, np.float64: 3e-14},
        atol={np.float32: 3e-4, np.float64: 3e-14})

  @jtu.sample_product(
    shape=[(15,), (3, 15), (1, 12)],
    dtype=jtu.dtypes.floating,
  )
  def testKdeIntegrateGaussian(self, shape, dtype):
    def scipy_fun(dataset, weights):
      kde = osp_stats.gaussian_kde(dataset, weights=np.abs(weights))
      # Note: the scipy implementation _always_ returns float64
      return kde.integrate_gaussian(mean, covariance).astype(dtype)

    def lax_fun(dataset, weights):
      kde = lsp_stats.gaussian_kde(dataset, weights=jax.numpy.abs(weights))
      return kde.integrate_gaussian(mean, covariance)

    # Construct a random mean and positive definite covariance matrix
    rng = jtu.rand_default(self.rng())
    ndim = shape[0] if len(shape) > 1 else 1
    mean = rng(ndim, dtype)
    L = rng((ndim, ndim), dtype)
    L[np.triu_indices(ndim, 1)] = 0.0
    L[np.diag_indices(ndim)] = np.exp(np.diag(L)) + 0.01
    covariance = L @ L.T

    args_maker = lambda: [
      rng(shape, dtype), rng(shape[-1:], dtype)]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                            tol={np.float32: 1e-3, np.float64: 1e-14})
    self._CompileAndCheck(
        lax_fun, args_maker, rtol={np.float32: 3e-07, np.float64: 4e-15})

  @jtu.sample_product(
    shape=[(15,), (12,)],
    dtype=jtu.dtypes.floating,
  )
  def testKdeIntegrateBox1d(self, shape, dtype):
    def scipy_fun(dataset, weights):
      kde = osp_stats.gaussian_kde(dataset, weights=np.abs(weights))
      # Note: the scipy implementation _always_ returns float64
      return kde.integrate_box_1d(-0.5, 1.5).astype(dtype)

    def lax_fun(dataset, weights):
      kde = lsp_stats.gaussian_kde(dataset, weights=jax.numpy.abs(weights))
      return kde.integrate_box_1d(-0.5, 1.5)

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [
      rng(shape, dtype), rng(shape[-1:], dtype)]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                            tol={np.float32: 1e-3, np.float64: 1e-14})
    self._CompileAndCheck(
        lax_fun, args_maker, rtol={np.float32: 3e-07, np.float64: 4e-15})

  @jtu.sample_product(
    shape=[(15,), (3, 15), (1, 12)],
    dtype=jtu.dtypes.floating,
  )
  def testKdeIntegrateKde(self, shape, dtype):
    def scipy_fun(dataset, weights):
      kde = osp_stats.gaussian_kde(dataset, weights=np.abs(weights))
      other = osp_stats.gaussian_kde(
        dataset[..., :-3] + 0.1, weights=np.abs(weights[:-3]))
      # Note: the scipy implementation _always_ returns float64
      return kde.integrate_kde(other).astype(dtype)

    def lax_fun(dataset, weights):
      kde = lsp_stats.gaussian_kde(dataset, weights=jax.numpy.abs(weights))
      other = lsp_stats.gaussian_kde(
        dataset[..., :-3] + 0.1, weights=jax.numpy.abs(weights[:-3]))
      return kde.integrate_kde(other)

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [
      rng(shape, dtype), rng(shape[-1:], dtype)]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                            tol={np.float32: 1e-3, np.float64: 1e-14})
    self._CompileAndCheck(
        lax_fun, args_maker, rtol={np.float32: 3e-07, np.float64: 4e-15})

  @jtu.sample_product(
    shape=[(15,), (3, 15), (1, 12)],
    dtype=jtu.dtypes.floating,
  )
  @jax.legacy_prng_key('allow')
  def testKdeResampleShape(self, shape, dtype):
    def resample(key, dataset, weights, *, shape):
      kde = lsp_stats.gaussian_kde(dataset, weights=jax.numpy.abs(weights))
      return kde.resample(key, shape=shape)

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [
      jax.random.PRNGKey(0), rng(shape, dtype), rng(shape[-1:], dtype)]

    ndim = shape[0] if len(shape) > 1 else 1

    func = partial(resample, shape=())
    with jax.debug_key_reuse(False):
      self._CompileAndCheck(
        func, args_maker, rtol={np.float32: 3e-07, np.float64: 4e-15})
    result = func(*args_maker())
    assert result.shape == (ndim,)

    func = partial(resample, shape=(4,))
    with jax.debug_key_reuse(False):
      self._CompileAndCheck(
        func, args_maker, rtol={np.float32: 3e-07, np.float64: 4e-15})
    result = func(*args_maker())
    assert result.shape == (ndim, 4)

  @jtu.sample_product(
    shape=[(15,), (1, 12)],
    dtype=jtu.dtypes.floating,
  )
  @jax.legacy_prng_key('allow')
  def testKdeResample1d(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    dataset = rng(shape, dtype)
    weights = jax.numpy.abs(rng(shape[-1:], dtype))
    kde = lsp_stats.gaussian_kde(dataset, weights=weights)
    samples = jax.numpy.squeeze(kde.resample(jax.random.PRNGKey(5), shape=(1000,)))

    def cdf(x):
      result = jax.vmap(partial(kde.integrate_box_1d, -np.inf))(x)
      # Manually casting to numpy in order to avoid type promotion error
      return np.array(result)

    self.assertGreater(osp_stats.kstest(samples, cdf).pvalue, 0.01)

  def testKdePyTree(self):
    @jax.jit
    def evaluate_kde(kde, x):
      return kde.evaluate(x)

    dtype = np.float32
    rng = jtu.rand_default(self.rng())
    dataset = rng((3, 15), dtype)
    x = rng((3, 12), dtype)
    kde = lsp_stats.gaussian_kde(dataset)
    leaves, treedef = jax.tree.flatten(kde)
    kde2 = jax.tree.unflatten(treedef, leaves)
    jax.tree.map(lambda a, b: self.assertAllClose(a, b), kde, kde2)
    self.assertAllClose(evaluate_kde(kde, x), kde.evaluate(x))

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape, axis in (
        ((0,), None),
        ((0,), 0),
        ((7,), None),
        ((7,), 0),
        ((47, 8), None),
        ((47, 8), 0),
        ((47, 8), 1),
        ((0, 2, 3), None),
        ((0, 2, 3), 0),
        ((0, 2, 3), 1),
        ((0, 2, 3), 2),
        ((10, 5, 21), None),
        ((10, 5, 21), 0),
        ((10, 5, 21), 1),
        ((10, 5, 21), 2),
      )
    ],
    dtype=jtu.dtypes.integer + jtu.dtypes.floating,
    contains_nans=[True, False],
    keepdims=[True, False]
  )
  def testMode(self, shape, dtype, axis, contains_nans, keepdims):
    if scipy_version < (1, 9, 0) and keepdims != True:
      self.skipTest("scipy < 1.9.0 only support keepdims == True")

    if contains_nans:
      rng = jtu.rand_some_nan(self.rng())
    else:
      rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    def scipy_mode_wrapper(a, axis=0, nan_policy='propagate', keepdims=None):
      """Wrapper to manage the shape discrepancies between scipy and jax"""
      if scipy_version < (1, 11, 0) and a.size == 0:
        if keepdims:
          if axis == None:
            output_shape = tuple(1 for _ in a.shape)
          else:
            output_shape = tuple(1 if i == axis else s for i, s in enumerate(a.shape))
        else:
          if axis == None:
            output_shape = ()
          else:
            output_shape = np.delete(np.array(a.shape, dtype=np.int64), axis)
        t = dtypes.canonicalize_dtype(jax.numpy.float_)
        return (np.full(output_shape, np.nan, dtype=t),
                np.zeros(output_shape, dtype=t))

      if scipy_version < (1, 9, 0):
        result = osp_stats.mode(a, axis=axis, nan_policy=nan_policy)
      else:
        result = osp_stats.mode(a, axis=axis, nan_policy=nan_policy, keepdims=keepdims)

      if a.size != 0 and axis == None and keepdims == True:
        output_shape = tuple(1 for _ in a.shape)
        return (result.mode.reshape(output_shape), result.count.reshape(output_shape))
      return result

    scipy_fun = partial(scipy_mode_wrapper, axis=axis, keepdims=keepdims)
    scipy_fun = jtu.ignore_warning(category=RuntimeWarning,
                                   message="Mean of empty slice.*")(scipy_fun)
    scipy_fun = jtu.ignore_warning(category=RuntimeWarning,
                                   message="invalid value encountered.*")(scipy_fun)
    lax_fun = partial(lsp_stats.mode, axis=axis, keepdims=keepdims)
    tol_spec = {np.float32: 2e-4, np.float64: 5e-6}
    tol = jtu.tolerance(dtype, tol_spec)
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(lax_fun, args_maker, rtol=tol)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in [(0,), (7,), (47, 8), (0, 2, 3), (10, 5, 21)]
      for axis in [None, *range(len(shape))
    ]],
    dtype=jtu.dtypes.integer + jtu.dtypes.floating,
    method=['average', 'min', 'max', 'dense', 'ordinal']
  )
  def testRankData(self, shape, dtype, axis, method):

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    scipy_fun = partial(osp_stats.rankdata, method=method, axis=axis)
    lax_fun = partial(lsp_stats.rankdata, method=method, axis=axis)
    tol_spec = {np.float32: 2e-4, np.float64: 5e-6}
    tol = jtu.tolerance(dtype, tol_spec)
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(lax_fun, args_maker, rtol=tol)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis, ddof=ddof, nan_policy=nan_policy, keepdims=keepdims)
      for shape in [(5,), (5, 6), (5, 6, 7)]
      for axis in [None, *range(len(shape))]
      for ddof in [0, 1, 2, 3]
      for nan_policy in ["propagate", "omit"]
      for keepdims in [True, False]
    ],
    dtype=jtu.dtypes.integer + jtu.dtypes.floating,
  )
  def testSEM(self, shape, dtype, axis, ddof, nan_policy, keepdims):

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    kwds = {} if scipy_version < (1, 11) else {'keepdims': keepdims}
    scipy_fun = partial(osp_stats.sem, axis=axis, ddof=ddof, nan_policy=nan_policy,
                        **kwds)
    lax_fun = partial(lsp_stats.sem, axis=axis, ddof=ddof, nan_policy=nan_policy,
                      **kwds)
    tol_spec = {np.float32: 2e-4, np.float64: 5e-6}
    tol = jtu.tolerance(dtype, tol_spec)
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            atol=tol)
    self._CompileAndCheck(lax_fun, args_maker, atol=tol)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
