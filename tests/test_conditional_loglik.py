"""Tests for _conditional_loglik, focused on ZINB2 branch."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import gammaln

from interlace.glmm_family import (
    BetaFamily,
    NegativeBinomial2Family,
    PoissonFamily,
    ZeroInflatedNB2Family,
    ZeroInflatedPoissonFamily,
    ZeroOneInflatedBetaFamily,
)
from interlace.glmm_laplace import _conditional_loglik


class TestConditionalLoglikZINB2:
    """Tests for the zeroinflated_negativebinomial branch of _conditional_loglik."""

    def test_hand_computed_values(self):
        """ZINB2 log-likelihood for y=[0, 3], mu=[2, 2], theta=1, pi=0.3.

        y=0: log[pi + (1-pi) * (theta/(mu+theta))^theta]
            = log[0.3 + 0.7 * (1/3)^1]
            = log[0.3 + 0.7/3]
            = log[0.53333...]

        y=3: log(1-pi) + NB2_logpmf(3, mu=2, theta=1)
            NB2_logpmf = lgamma(y+theta) - lgamma(theta) - lgamma(y+1)
                       + theta*log(theta/(mu+theta)) + y*log(mu/(mu+theta))
                       = lgamma(4) - lgamma(1) - lgamma(4)
                       + 1*log(1/3) + 3*log(2/3)
                       = 0 + log(1/3) + 3*log(2/3)
            ll_y3 = log(0.7) + log(1/3) + 3*log(2/3)

        Total = ll_y0 + ll_y3
        """
        theta = 1.0
        pi = 0.3
        fam = ZeroInflatedNB2Family(theta=theta, pi=pi)
        y = np.array([0.0, 3.0])
        mu = np.array([2.0, 2.0])
        wt = np.array([1.0, 1.0])

        # y=0 term
        nb2_pmf_at_zero = (theta / (mu[0] + theta)) ** theta  # (1/3)^1
        ll_y0 = np.log(pi + (1 - pi) * nb2_pmf_at_zero)

        # y=3 term
        nb2_ll = (
            gammaln(3.0 + theta)
            - gammaln(theta)
            - gammaln(3.0 + 1)
            + theta * np.log(theta / (mu[1] + theta))
            + 3.0 * np.log(mu[1] / (mu[1] + theta))
        )
        ll_y3 = np.log(1 - pi) + nb2_ll

        expected = ll_y0 + ll_y3
        result = _conditional_loglik(y, mu, wt, fam)
        assert_allclose(result, expected, atol=1e-12)

    def test_pi_zero_matches_nb2(self):
        """With pi=0, ZINB2 log-likelihood should equal NB2 log-likelihood."""
        theta = 2.0
        fam_zinb = ZeroInflatedNB2Family(theta=theta, pi=0.0)
        fam_nb2 = NegativeBinomial2Family(theta=theta)
        y = np.array([0.0, 1.0, 3.0, 7.0, 0.0])
        mu = np.array([2.0, 2.0, 2.0, 5.0, 0.5])
        wt = np.ones(5)

        ll_zinb = _conditional_loglik(y, mu, wt, fam_zinb)
        ll_nb2 = _conditional_loglik(y, mu, wt, fam_nb2)
        assert_allclose(ll_zinb, ll_nb2, atol=1e-12)

    def test_all_zeros(self):
        """All-zero data with zero-inflation should have higher likelihood
        than without it (more probability mass at zero)."""
        theta = 1.0
        y = np.array([0.0, 0.0, 0.0, 0.0])
        mu = np.array([3.0, 3.0, 3.0, 3.0])
        wt = np.ones(4)

        fam_no_zi = ZeroInflatedNB2Family(theta=theta, pi=0.0)
        fam_with_zi = ZeroInflatedNB2Family(theta=theta, pi=0.5)

        ll_no_zi = _conditional_loglik(y, mu, wt, fam_no_zi)
        ll_with_zi = _conditional_loglik(y, mu, wt, fam_with_zi)
        assert ll_with_zi > ll_no_zi

    def test_weights_scale_loglik(self):
        """Doubling weights should double the positive-count contributions."""
        theta = 1.5
        pi = 0.2
        fam = ZeroInflatedNB2Family(theta=theta, pi=pi)
        # Use only positive counts so weight scaling is clean
        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([2.0, 2.0, 4.0])
        wt1 = np.ones(3)
        wt2 = 2.0 * np.ones(3)

        ll1 = _conditional_loglik(y, mu, wt1, fam)
        ll2 = _conditional_loglik(y, mu, wt2, fam)
        assert_allclose(ll2, 2.0 * ll1, atol=1e-12)

    def test_large_pi_concentrates_at_zero(self):
        """With pi near 1, log-likelihood for y=0 approaches log(1) = 0."""
        theta = 1.0
        fam = ZeroInflatedNB2Family(theta=theta, pi=0.999)
        y = np.array([0.0])
        mu = np.array([5.0])
        wt = np.array([1.0])

        ll = _conditional_loglik(y, mu, wt, fam)
        # log(0.999 + 0.001 * tiny) ≈ log(0.999) ≈ -0.001
        assert ll > -0.01  # very close to zero

    def test_returns_finite(self):
        """Log-likelihood should be finite for reasonable inputs."""
        theta = 1.0
        pi = 0.3
        fam = ZeroInflatedNB2Family(theta=theta, pi=pi)
        rng = np.random.default_rng(42)
        y = rng.negative_binomial(n=2, p=0.5, size=100).astype(float)
        mu = np.full(100, 3.0)
        wt = np.ones(100)

        ll = _conditional_loglik(y, mu, wt, fam)
        assert np.isfinite(ll)


class TestConditionalLoglikZIP:
    """Tests for the zeroinflated_poisson branch of _conditional_loglik."""

    def test_hand_computed_values(self):
        """ZIP log-likelihood for y=[0, 3], mu=[2, 2], pi=0.3.

        y=0: log[pi + (1-pi) * exp(-mu)]
            = log[0.3 + 0.7 * exp(-2)]
            = log[0.3 + 0.7 * 0.13533528...]
            = log[0.39473470...]

        y=3: log(1-pi) + Poisson_logpmf(3, mu=2)
            Poisson_logpmf = y*log(mu) - mu - lgamma(y+1)
                           = 3*log(2) - 2 - lgamma(4)
                           = 3*log(2) - 2 - log(6)
            ll_y3 = log(0.7) + 3*log(2) - 2 - log(6)

        Total = ll_y0 + ll_y3
        """
        pi = 0.3
        fam = ZeroInflatedPoissonFamily(pi=pi)
        y = np.array([0.0, 3.0])
        mu = np.array([2.0, 2.0])
        wt = np.array([1.0, 1.0])

        # y=0 term
        ll_y0 = np.log(pi + (1 - pi) * np.exp(-mu[0]))

        # y=3 term
        pois_ll = 3.0 * np.log(mu[1]) - mu[1] - gammaln(3.0 + 1)
        ll_y3 = np.log(1 - pi) + pois_ll

        expected = ll_y0 + ll_y3
        result = _conditional_loglik(y, mu, wt, fam)
        assert_allclose(result, expected, atol=1e-12)

    def test_pi_zero_matches_poisson(self):
        """With pi=0, ZIP log-likelihood should equal Poisson log-likelihood."""
        fam_zip = ZeroInflatedPoissonFamily(pi=0.0)
        fam_pois = PoissonFamily()
        y = np.array([0.0, 1.0, 3.0, 7.0, 0.0])
        mu = np.array([2.0, 2.0, 2.0, 5.0, 0.5])
        wt = np.ones(5)

        ll_zip = _conditional_loglik(y, mu, wt, fam_zip)
        ll_pois = _conditional_loglik(y, mu, wt, fam_pois)
        assert_allclose(ll_zip, ll_pois, atol=1e-12)

    def test_all_zeros(self):
        """All-zero data with zero-inflation should have higher likelihood
        than without it (more probability mass at zero)."""
        y = np.array([0.0, 0.0, 0.0, 0.0])
        mu = np.array([3.0, 3.0, 3.0, 3.0])
        wt = np.ones(4)

        fam_no_zi = ZeroInflatedPoissonFamily(pi=0.0)
        fam_with_zi = ZeroInflatedPoissonFamily(pi=0.5)

        ll_no_zi = _conditional_loglik(y, mu, wt, fam_no_zi)
        ll_with_zi = _conditional_loglik(y, mu, wt, fam_with_zi)
        assert ll_with_zi > ll_no_zi

    def test_weights_scale_loglik(self):
        """Doubling weights should double the positive-count contributions."""
        pi = 0.2
        fam = ZeroInflatedPoissonFamily(pi=pi)
        # Use only positive counts so weight scaling is clean
        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([2.0, 2.0, 4.0])
        wt1 = np.ones(3)
        wt2 = 2.0 * np.ones(3)

        ll1 = _conditional_loglik(y, mu, wt1, fam)
        ll2 = _conditional_loglik(y, mu, wt2, fam)
        assert_allclose(ll2, 2.0 * ll1, atol=1e-12)

    def test_large_pi_concentrates_at_zero(self):
        """With pi near 1, log-likelihood for y=0 approaches log(1) = 0."""
        fam = ZeroInflatedPoissonFamily(pi=0.999)
        y = np.array([0.0])
        mu = np.array([5.0])
        wt = np.array([1.0])

        ll = _conditional_loglik(y, mu, wt, fam)
        # log(0.999 + 0.001 * tiny) ≈ log(0.999) ≈ -0.001
        assert ll > -0.01

    def test_returns_finite(self):
        """Log-likelihood should be finite for reasonable inputs."""
        pi = 0.3
        fam = ZeroInflatedPoissonFamily(pi=pi)
        rng = np.random.default_rng(42)
        y = rng.poisson(lam=3.0, size=100).astype(float)
        mu = np.full(100, 3.0)
        wt = np.ones(100)

        ll = _conditional_loglik(y, mu, wt, fam)
        assert np.isfinite(ll)


class TestConditionalLoglikZOIB:
    """Tests for the zerooneinflated_beta branch of _conditional_loglik."""

    def test_hand_computed_interior(self):
        """For interior observations (0 < y < 1), ZOIB log-likelihood is
        log(p_beta) + Beta_logpdf, where p_beta = 1 - p0 - p1."""
        phi = 3.0
        p0, p1 = 0.2, 0.1
        p_beta = 1.0 - p0 - p1
        mu_val = 0.4

        y_int = np.array([0.5])
        mu_int = np.array([mu_val])
        wt_int = np.array([1.0])

        # Without inflation: pure Beta logpdf
        fam_noinfl = ZeroOneInflatedBetaFamily(phi=phi, p0=0.0, p1=0.0)
        ll_noinfl = _conditional_loglik(y_int, mu_int, wt_int, fam_noinfl)
        expected_interior = ll_noinfl + np.log(p_beta)

        # With inflation: log(p_beta) + Beta logpdf
        fam_full = ZeroOneInflatedBetaFamily(phi=phi, p0=p0, p1=p1)
        ll_full = _conditional_loglik(y_int, mu_int, wt_int, fam_full)
        assert_allclose(ll_full, expected_interior, atol=1e-12)

    def test_mixed_boundary_and_interior(self):
        """Log-likelihood is finite for data with 0, interior, and 1 values."""
        phi = 3.0
        fam = ZeroOneInflatedBetaFamily(phi=phi, p0=0.2, p1=0.1)
        y = np.array([0.0, 0.5, 1.0])
        mu = np.full(3, 0.4)
        wt = np.ones(3)

        result = _conditional_loglik(y, mu, wt, fam)
        assert np.isfinite(result)

    def test_no_inflation_matches_beta(self):
        """With p0=p1=0, ZOIB log-likelihood should equal Beta log-likelihood."""
        phi = 5.0
        fam_zoib = ZeroOneInflatedBetaFamily(phi=phi, p0=0.0, p1=0.0)
        fam_beta = BetaFamily(phi=phi)
        y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mu = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
        wt = np.ones(5)

        ll_zoib = _conditional_loglik(y, mu, wt, fam_zoib)
        ll_beta = _conditional_loglik(y, mu, wt, fam_beta)
        assert_allclose(ll_zoib, ll_beta, atol=1e-12)

    def test_boundary_values_higher_with_inflation(self):
        """Data with 0s and 1s should have higher likelihood when inflation
        probabilities are set appropriately."""
        phi = 5.0
        y = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        mu = np.array([0.3, 0.3, 0.5, 0.7, 0.7])
        wt = np.ones(5)

        fam_no_infl = ZeroOneInflatedBetaFamily(phi=phi, p0=0.0, p1=0.0)
        fam_with_infl = ZeroOneInflatedBetaFamily(phi=phi, p0=0.3, p1=0.3)

        ll_no = _conditional_loglik(y, mu, wt, fam_no_infl)
        ll_with = _conditional_loglik(y, mu, wt, fam_with_infl)
        assert ll_with > ll_no

    def test_zero_only_inflation(self):
        """With p1=0, only zero-inflation is active."""
        phi = 3.0
        p0 = 0.4
        fam = ZeroOneInflatedBetaFamily(phi=phi, p0=p0, p1=0.0)
        y = np.array([0.0, 0.0, 0.5])
        mu = np.array([0.5, 0.5, 0.5])
        wt = np.ones(3)

        ll = _conditional_loglik(y, mu, wt, fam)
        assert np.isfinite(ll)

    def test_one_only_inflation(self):
        """With p0=0, only one-inflation is active."""
        phi = 3.0
        p1 = 0.4
        fam = ZeroOneInflatedBetaFamily(phi=phi, p0=0.0, p1=p1)
        y = np.array([0.5, 1.0, 1.0])
        mu = np.array([0.5, 0.5, 0.5])
        wt = np.ones(3)

        ll = _conditional_loglik(y, mu, wt, fam)
        assert np.isfinite(ll)

    def test_weights_scale_loglik(self):
        """Doubling weights should double the log-likelihood for interior obs."""
        phi = 3.0
        fam = ZeroOneInflatedBetaFamily(phi=phi, p0=0.1, p1=0.1)
        y = np.array([0.2, 0.5, 0.8])
        mu = np.array([0.3, 0.5, 0.7])
        wt1 = np.ones(3)
        wt2 = 2.0 * np.ones(3)

        ll1 = _conditional_loglik(y, mu, wt1, fam)
        ll2 = _conditional_loglik(y, mu, wt2, fam)
        assert_allclose(ll2, 2.0 * ll1, atol=1e-12)

    def test_returns_finite(self):
        """Log-likelihood should be finite for reasonable inputs."""
        phi = 5.0
        fam = ZeroOneInflatedBetaFamily(phi=phi, p0=0.1, p1=0.05)
        rng = np.random.default_rng(42)
        y = rng.beta(2, 5, size=100)
        # Add some boundary values
        y[0] = 0.0
        y[1] = 1.0
        mu = np.clip(rng.beta(2, 5, size=100), 0.01, 0.99)
        wt = np.ones(100)

        ll = _conditional_loglik(y, mu, wt, fam)
        assert np.isfinite(ll)
