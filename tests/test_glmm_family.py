"""Tests for GLMMFamily protocol and concrete family implementations."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from interlace.glmm_family import (
    BetaFamily,
    BinomialFamily,
    GaussianFamily,
    GLMMFamily,
    NegativeBinomial2Family,
    PoissonFamily,
    ZeroInflatedNB2Family,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eta():
    """Linear predictor values spanning a reasonable range."""
    return np.array([-3.0, -1.0, 0.0, 1.0, 3.0])


@pytest.fixture
def prob():
    """Probabilities for binomial tests."""
    return np.array([0.01, 0.25, 0.5, 0.75, 0.99])


@pytest.fixture
def counts():
    """Positive counts for Poisson tests."""
    return np.array([0.5, 1.0, 2.0, 5.0, 10.0])


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Every concrete family must satisfy the GLMMFamily protocol."""

    @pytest.mark.parametrize(
        "cls",
        [
            BinomialFamily,
            PoissonFamily,
            GaussianFamily,
            NegativeBinomial2Family,
            BetaFamily,
            ZeroInflatedNB2Family,
        ],
    )
    def test_is_runtime_checkable(self, cls):
        family = cls()
        assert isinstance(family, GLMMFamily)

    @pytest.mark.parametrize(
        "cls",
        [
            BinomialFamily,
            PoissonFamily,
            GaussianFamily,
            NegativeBinomial2Family,
            BetaFamily,
            ZeroInflatedNB2Family,
        ],
    )
    def test_has_required_attributes(self, cls):
        family = cls()
        for attr in ("link", "linkinv", "mu_eta", "variance", "dev_resids", "name"):
            assert hasattr(family, attr), f"Missing {attr}"


# ---------------------------------------------------------------------------
# Binomial (logit link)
# ---------------------------------------------------------------------------


class TestBinomialFamily:
    def test_link_linkinv_roundtrip(self, prob):
        """link(linkinv(eta)) == eta and linkinv(link(mu)) == mu."""
        fam = BinomialFamily()
        eta = fam.link(prob)
        assert_allclose(fam.linkinv(eta), prob, atol=1e-12)
        # And the other direction
        mu_back = fam.linkinv(fam.link(prob))
        assert_allclose(mu_back, prob, atol=1e-12)

    def test_link_known_values(self):
        """logit(0.5) = 0, logit(0.25) = -log(3)."""
        fam = BinomialFamily()
        assert_allclose(fam.link(np.array([0.5])), [0.0], atol=1e-12)
        assert_allclose(fam.link(np.array([0.25])), [-np.log(3)], atol=1e-12)

    def test_linkinv_known_values(self):
        """expit(0) = 0.5."""
        fam = BinomialFamily()
        assert_allclose(fam.linkinv(np.array([0.0])), [0.5], atol=1e-12)

    def test_variance(self, prob):
        """Var(mu) = mu * (1 - mu) for binomial."""
        fam = BinomialFamily()
        expected = prob * (1.0 - prob)
        assert_allclose(fam.variance(prob), expected, atol=1e-12)

    def test_mu_eta_is_derivative(self):
        """mu_eta should equal d(linkinv)/d(eta), verified numerically."""
        fam = BinomialFamily()
        eta = np.array([-2.0, 0.0, 2.0])
        h = 1e-7
        numerical = (fam.linkinv(eta + h) - fam.linkinv(eta - h)) / (2 * h)
        assert_allclose(fam.mu_eta(eta), numerical, rtol=1e-5)

    def test_dev_resids_perfect_fit(self):
        """Deviance residuals should be zero when y == mu."""
        fam = BinomialFamily()
        mu = np.array([0.3, 0.5, 0.7])
        wt = np.ones_like(mu)
        assert_allclose(fam.dev_resids(mu, mu, wt), 0.0, atol=1e-12)

    def test_dev_resids_known_value(self):
        """Check deviance residual against hand-computed value.

        For y=1, mu=0.5, wt=1:
          d_i = 2 * [y*log(y/mu) + (1-y)*log((1-y)/(1-mu))]
              = 2 * [1*log(2) + 0]  = 2*log(2)
        """
        fam = BinomialFamily()
        y = np.array([1.0])
        mu = np.array([0.5])
        wt = np.array([1.0])
        expected = 2.0 * np.log(2.0)
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-12)

    def test_name(self):
        fam = BinomialFamily()
        assert fam.name == "binomial"

    def test_linkinv_extreme_eta(self):
        """Should not overflow for very large/small eta."""
        fam = BinomialFamily()
        eta = np.array([-500.0, 500.0])
        mu = fam.linkinv(eta)
        assert np.all(np.isfinite(mu))
        assert mu[0] < 1e-10
        assert mu[1] > 1.0 - 1e-10


# ---------------------------------------------------------------------------
# Poisson (log link)
# ---------------------------------------------------------------------------


class TestPoissonFamily:
    def test_link_linkinv_roundtrip(self, counts):
        """link(linkinv(eta)) == eta."""
        fam = PoissonFamily()
        eta = fam.link(counts)
        assert_allclose(fam.linkinv(eta), counts, atol=1e-12)

    def test_link_known_values(self):
        """log(1) = 0, log(e) = 1."""
        fam = PoissonFamily()
        assert_allclose(fam.link(np.array([1.0])), [0.0], atol=1e-12)
        assert_allclose(fam.link(np.array([np.e])), [1.0], atol=1e-12)

    def test_linkinv_known_values(self):
        """exp(0) = 1."""
        fam = PoissonFamily()
        assert_allclose(fam.linkinv(np.array([0.0])), [1.0], atol=1e-12)

    def test_variance(self, counts):
        """Var(mu) = mu for Poisson."""
        fam = PoissonFamily()
        assert_allclose(fam.variance(counts), counts, atol=1e-12)

    def test_mu_eta_is_derivative(self):
        """mu_eta = d(exp(eta))/d(eta) = exp(eta)."""
        fam = PoissonFamily()
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        assert_allclose(fam.mu_eta(eta), np.exp(eta), atol=1e-12)

    def test_dev_resids_perfect_fit(self):
        """Deviance residuals should be zero when y == mu."""
        fam = PoissonFamily()
        mu = np.array([1.0, 2.0, 5.0])
        wt = np.ones_like(mu)
        assert_allclose(fam.dev_resids(mu, mu, wt), 0.0, atol=1e-12)

    def test_dev_resids_known_value(self):
        """For y=3, mu=1, wt=1:
        d_i = 2 * (y*log(y/mu) - (y - mu))
            = 2 * (3*log(3) - 2)
        """
        fam = PoissonFamily()
        y = np.array([3.0])
        mu = np.array([1.0])
        wt = np.array([1.0])
        expected = 2.0 * (3.0 * np.log(3.0) - 2.0)
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-12)

    def test_dev_resids_y_zero(self):
        """For y=0, mu=2, wt=1: d_i = 2 * mu = 4."""
        fam = PoissonFamily()
        y = np.array([0.0])
        mu = np.array([2.0])
        wt = np.array([1.0])
        expected = np.array([4.0])
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-12)

    def test_name(self):
        fam = PoissonFamily()
        assert fam.name == "poisson"

    def test_linkinv_clamps_large_eta(self):
        """exp(1000) would overflow; linkinv should stay finite."""
        fam = PoissonFamily()
        mu = fam.linkinv(np.array([500.0]))
        assert np.all(np.isfinite(mu))


# ---------------------------------------------------------------------------
# Gaussian (identity link)
# ---------------------------------------------------------------------------


class TestGaussianFamily:
    def test_link_is_identity(self, eta):
        fam = GaussianFamily()
        assert_allclose(fam.link(eta), eta, atol=1e-12)

    def test_linkinv_is_identity(self, eta):
        fam = GaussianFamily()
        assert_allclose(fam.linkinv(eta), eta, atol=1e-12)

    def test_variance_is_one(self, eta):
        fam = GaussianFamily()
        assert_allclose(fam.variance(eta), np.ones_like(eta), atol=1e-12)

    def test_mu_eta_is_one(self, eta):
        fam = GaussianFamily()
        assert_allclose(fam.mu_eta(eta), np.ones_like(eta), atol=1e-12)

    def test_dev_resids(self):
        """Gaussian deviance residual = wt * (y - mu)^2."""
        fam = GaussianFamily()
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 1.5, 1.5])
        wt = np.array([1.0, 2.0, 1.0])
        expected = wt * (y - mu) ** 2
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-12)

    def test_name(self):
        fam = GaussianFamily()
        assert fam.name == "gaussian"


# ---------------------------------------------------------------------------
# Negative Binomial 2 (log link)
# ---------------------------------------------------------------------------


class TestNegativeBinomial2Family:
    def test_name(self):
        fam = NegativeBinomial2Family(theta=1.0)
        assert fam.name == "negativebinomial"

    def test_default_theta(self):
        fam = NegativeBinomial2Family()
        assert fam.theta == 1.0

    def test_custom_theta(self):
        fam = NegativeBinomial2Family(theta=2.5)
        assert fam.theta == 2.5

    def test_link_linkinv_roundtrip(self, counts):
        """link(linkinv(eta)) == eta."""
        fam = NegativeBinomial2Family(theta=1.0)
        eta = fam.link(counts)
        assert_allclose(fam.linkinv(eta), counts, atol=1e-12)

    def test_link_known_values(self):
        """log(1) = 0, log(e) = 1."""
        fam = NegativeBinomial2Family(theta=1.0)
        assert_allclose(fam.link(np.array([1.0])), [0.0], atol=1e-12)
        assert_allclose(fam.link(np.array([np.e])), [1.0], atol=1e-12)

    def test_linkinv_known_values(self):
        """exp(0) = 1."""
        fam = NegativeBinomial2Family(theta=1.0)
        assert_allclose(fam.linkinv(np.array([0.0])), [1.0], atol=1e-12)

    def test_variance_formula(self, counts):
        """Var(mu) = mu + mu^2 / theta."""
        theta = 2.0
        fam = NegativeBinomial2Family(theta=theta)
        expected = counts + counts**2 / theta
        assert_allclose(fam.variance(counts), expected, atol=1e-12)

    def test_variance_reduces_to_poisson_large_theta(self, counts):
        """As theta -> inf, NB2 variance -> mu (Poisson)."""
        fam = NegativeBinomial2Family(theta=1e12)
        assert_allclose(fam.variance(counts), counts, rtol=1e-6)

    def test_mu_eta_is_derivative(self):
        """mu_eta should equal d(linkinv)/d(eta), verified numerically."""
        fam = NegativeBinomial2Family(theta=1.5)
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        h = 1e-7
        numerical = (fam.linkinv(eta + h) - fam.linkinv(eta - h)) / (2 * h)
        assert_allclose(fam.mu_eta(eta), numerical, rtol=1e-5)

    def test_mu_eta_equals_exp(self):
        """For log link, d(exp(eta))/d(eta) = exp(eta)."""
        fam = NegativeBinomial2Family(theta=1.0)
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        assert_allclose(fam.mu_eta(eta), np.exp(eta), atol=1e-12)

    def test_dev_resids_perfect_fit(self):
        """Deviance residuals should be zero when y == mu."""
        fam = NegativeBinomial2Family(theta=1.0)
        mu = np.array([1.0, 2.0, 5.0])
        wt = np.ones_like(mu)
        assert_allclose(fam.dev_resids(mu, mu, wt), 0.0, atol=1e-12)

    def test_dev_resids_known_value(self):
        """NB2 deviance for y=3, mu=1, theta=1, wt=1.

        d_i = 2 * wt * [y*log(y/mu) - (y + theta)*log((y + theta)/(mu + theta))]
             = 2 * [3*log(3) - 4*log(4/2)]
             = 2 * [3*log(3) - 4*log(2)]
        """
        fam = NegativeBinomial2Family(theta=1.0)
        y = np.array([3.0])
        mu = np.array([1.0])
        wt = np.array([1.0])
        expected = 2.0 * (3.0 * np.log(3.0) - 4.0 * np.log(2.0))
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-12)

    def test_dev_resids_y_zero(self):
        """For y=0, mu=2, theta=1, wt=1:
        d_i = 2 * [0*log(0/2) - (0+1)*log((0+1)/(2+1))]
            = 2 * [0 - log(1/3)]
            = 2 * log(3)
        """
        fam = NegativeBinomial2Family(theta=1.0)
        y = np.array([0.0])
        mu = np.array([2.0])
        wt = np.array([1.0])
        expected = np.array([2.0 * np.log(3.0)])
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-12)

    def test_linkinv_clamps_large_eta(self):
        """exp(1000) would overflow; linkinv should stay finite."""
        fam = NegativeBinomial2Family(theta=1.0)
        mu = fam.linkinv(np.array([500.0]))
        assert np.all(np.isfinite(mu))

    def test_theta_must_be_positive(self):
        """theta <= 0 should raise."""
        with pytest.raises(ValueError, match="theta must be positive"):
            NegativeBinomial2Family(theta=0.0)
        with pytest.raises(ValueError, match="theta must be positive"):
            NegativeBinomial2Family(theta=-1.0)


# ---------------------------------------------------------------------------
# Zero-Inflated Negative Binomial 2 (log link, count component)
# ---------------------------------------------------------------------------


class TestZeroInflatedNB2Family:
    """Tests for ZeroInflatedNB2Family.

    The count component uses a log link and NB2 variance, identical to
    NegativeBinomial2Family.  The zero-inflation probability pi is stored
    on the family but does not alter link/linkinv/mu_eta/variance (those
    operate on the count linear predictor only).  dev_resids uses the
    count-component NB2 deviance.
    """

    def test_name(self):
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        assert fam.name == "zeroinflated_negativebinomial"

    def test_default_params(self):
        fam = ZeroInflatedNB2Family()
        assert fam.theta == 1.0
        assert fam.pi == 0.0

    def test_custom_theta(self):
        fam = ZeroInflatedNB2Family(theta=2.5)
        assert fam.theta == 2.5

    def test_custom_pi(self):
        fam = ZeroInflatedNB2Family(pi=0.3)
        assert fam.pi == 0.3

    def test_theta_must_be_positive(self):
        with pytest.raises(ValueError, match="theta must be positive"):
            ZeroInflatedNB2Family(theta=0.0)
        with pytest.raises(ValueError, match="theta must be positive"):
            ZeroInflatedNB2Family(theta=-1.0)

    def test_pi_must_be_in_unit_interval(self):
        """pi must be in [0, 1)."""
        with pytest.raises(ValueError, match="pi must be in"):
            ZeroInflatedNB2Family(pi=-0.1)
        with pytest.raises(ValueError, match="pi must be in"):
            ZeroInflatedNB2Family(pi=1.0)
        with pytest.raises(ValueError, match="pi must be in"):
            ZeroInflatedNB2Family(pi=1.5)
        # Edge: pi=0 is valid (no zero-inflation, reduces to NB2)
        fam = ZeroInflatedNB2Family(pi=0.0)
        assert fam.pi == 0.0

    # -- Link functions (log link, same as NB2) --

    def test_link_linkinv_roundtrip(self, counts):
        """link(linkinv(eta)) == eta."""
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        eta = fam.link(counts)
        assert_allclose(fam.linkinv(eta), counts, atol=1e-12)

    def test_link_known_values(self):
        """log(1) = 0, log(e) = 1."""
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        assert_allclose(fam.link(np.array([1.0])), [0.0], atol=1e-12)
        assert_allclose(fam.link(np.array([np.e])), [1.0], atol=1e-12)

    def test_linkinv_known_values(self):
        """exp(0) = 1."""
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        assert_allclose(fam.linkinv(np.array([0.0])), [1.0], atol=1e-12)

    def test_linkinv_clamps_large_eta(self):
        """exp(1000) would overflow; linkinv should stay finite."""
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        mu = fam.linkinv(np.array([500.0]))
        assert np.all(np.isfinite(mu))

    # -- Variance (NB2 count component) --

    def test_variance_formula(self, counts):
        """Var(mu) = mu + mu^2 / theta (count component only)."""
        theta = 2.0
        fam = ZeroInflatedNB2Family(theta=theta, pi=0.3)
        expected = counts + counts**2 / theta
        assert_allclose(fam.variance(counts), expected, atol=1e-12)

    def test_variance_independent_of_pi(self, counts):
        """Variance function does not depend on pi (count component only)."""
        fam0 = ZeroInflatedNB2Family(theta=2.0, pi=0.0)
        fam5 = ZeroInflatedNB2Family(theta=2.0, pi=0.5)
        assert_allclose(fam0.variance(counts), fam5.variance(counts), atol=1e-15)

    def test_variance_matches_nb2(self, counts):
        """Count-component variance should match NegativeBinomial2Family."""
        theta = 1.5
        fam_zinb = ZeroInflatedNB2Family(theta=theta, pi=0.3)
        fam_nb2 = NegativeBinomial2Family(theta=theta)
        assert_allclose(fam_zinb.variance(counts), fam_nb2.variance(counts), atol=1e-15)

    def test_variance_reduces_to_poisson_large_theta(self, counts):
        """As theta -> inf, count variance -> mu (Poisson)."""
        fam = ZeroInflatedNB2Family(theta=1e12, pi=0.2)
        assert_allclose(fam.variance(counts), counts, rtol=1e-6)

    # -- mu_eta (d(linkinv)/d(eta)) --

    def test_mu_eta_is_derivative(self):
        """mu_eta should equal d(linkinv)/d(eta), verified numerically."""
        fam = ZeroInflatedNB2Family(theta=1.5, pi=0.2)
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        h = 1e-7
        numerical = (fam.linkinv(eta + h) - fam.linkinv(eta - h)) / (2 * h)
        assert_allclose(fam.mu_eta(eta), numerical, rtol=1e-5)

    def test_mu_eta_equals_exp(self):
        """For log link, d(exp(eta))/d(eta) = exp(eta)."""
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        assert_allclose(fam.mu_eta(eta), np.exp(eta), atol=1e-12)

    # -- Deviance residuals (NB2 count component) --

    def test_dev_resids_perfect_fit(self):
        """Deviance residuals should be zero when y == mu."""
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        mu = np.array([1.0, 2.0, 5.0])
        wt = np.ones_like(mu)
        assert_allclose(fam.dev_resids(mu, mu, wt), 0.0, atol=1e-12)

    def test_dev_resids_known_value(self):
        """NB2 deviance for y=3, mu=1, theta=1, wt=1.

        d_i = 2 * wt * [y*log(y/mu) - (y + theta)*log((y + theta)/(mu + theta))]
             = 2 * [3*log(3) - 4*log(2)]
        """
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        y = np.array([3.0])
        mu = np.array([1.0])
        wt = np.array([1.0])
        expected = 2.0 * (3.0 * np.log(3.0) - 4.0 * np.log(2.0))
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-12)

    def test_dev_resids_y_zero(self):
        """For y=0, mu=2, theta=1, wt=1:
        d_i = 2 * [0*log(0/2) - (0+1)*log((0+1)/(2+1))]
            = 2 * log(3)
        """
        fam = ZeroInflatedNB2Family(theta=1.0, pi=0.2)
        y = np.array([0.0])
        mu = np.array([2.0])
        wt = np.array([1.0])
        expected = np.array([2.0 * np.log(3.0)])
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-12)

    def test_dev_resids_matches_nb2(self):
        """Count-component deviance should match NegativeBinomial2Family."""
        theta = 1.5
        fam_zinb = ZeroInflatedNB2Family(theta=theta, pi=0.3)
        fam_nb2 = NegativeBinomial2Family(theta=theta)
        y = np.array([0.0, 1.0, 3.0, 7.0])
        mu = np.array([2.0, 2.0, 2.0, 2.0])
        wt = np.ones(4)
        assert_allclose(
            fam_zinb.dev_resids(y, mu, wt),
            fam_nb2.dev_resids(y, mu, wt),
            atol=1e-12,
        )

    # -- pi=0 reduces to NB2 --

    def test_pi_zero_matches_nb2_everywhere(self, counts):
        """With pi=0, ZINB2 should be identical to NB2 for all methods."""
        theta = 2.0
        fam_zinb = ZeroInflatedNB2Family(theta=theta, pi=0.0)
        fam_nb2 = NegativeBinomial2Family(theta=theta)
        eta = np.log(counts)
        assert_allclose(fam_zinb.link(counts), fam_nb2.link(counts), atol=1e-15)
        assert_allclose(fam_zinb.linkinv(eta), fam_nb2.linkinv(eta), atol=1e-15)
        assert_allclose(fam_zinb.mu_eta(eta), fam_nb2.mu_eta(eta), atol=1e-15)
        assert_allclose(fam_zinb.variance(counts), fam_nb2.variance(counts), atol=1e-15)


# ---------------------------------------------------------------------------
# Beta (logit link)
# ---------------------------------------------------------------------------


class TestBetaFamily:
    def test_name(self):
        fam = BetaFamily()
        assert fam.name == "beta"

    def test_default_phi(self):
        fam = BetaFamily()
        assert fam.phi == 1.0

    def test_custom_phi(self):
        fam = BetaFamily(phi=5.0)
        assert fam.phi == 5.0

    def test_phi_must_be_positive(self):
        with pytest.raises(ValueError, match="phi must be positive"):
            BetaFamily(phi=0.0)
        with pytest.raises(ValueError, match="phi must be positive"):
            BetaFamily(phi=-1.0)

    def test_link_linkinv_roundtrip(self, prob):
        """link(linkinv(eta)) == eta and linkinv(link(mu)) == mu."""
        fam = BetaFamily(phi=5.0)
        eta = fam.link(prob)
        assert_allclose(fam.linkinv(eta), prob, atol=1e-12)
        mu_back = fam.linkinv(fam.link(prob))
        assert_allclose(mu_back, prob, atol=1e-12)

    def test_link_known_values(self):
        """logit(0.5) = 0, logit(0.25) = -log(3)."""
        fam = BetaFamily()
        assert_allclose(fam.link(np.array([0.5])), [0.0], atol=1e-12)
        assert_allclose(fam.link(np.array([0.25])), [-np.log(3)], atol=1e-12)

    def test_linkinv_known_values(self):
        """expit(0) = 0.5."""
        fam = BetaFamily()
        assert_allclose(fam.linkinv(np.array([0.0])), [0.5], atol=1e-12)

    def test_variance(self, prob):
        """Var(mu) = mu * (1 - mu) / (1 + phi)."""
        phi = 5.0
        fam = BetaFamily(phi=phi)
        expected = prob * (1.0 - prob) / (1.0 + phi)
        assert_allclose(fam.variance(prob), expected, atol=1e-12)

    def test_variance_reduces_to_binomial_small_phi(self, prob):
        """As phi -> 0+, Beta variance -> mu*(1-mu) (binomial-like)."""
        fam = BetaFamily(phi=1e-10)
        expected = prob * (1.0 - prob)
        assert_allclose(fam.variance(prob), expected, rtol=1e-6)

    def test_variance_shrinks_with_large_phi(self, prob):
        """Larger phi (more precision) means smaller variance."""
        fam_lo = BetaFamily(phi=1.0)
        fam_hi = BetaFamily(phi=100.0)
        assert np.all(fam_hi.variance(prob) < fam_lo.variance(prob))

    def test_mu_eta_is_derivative(self):
        """mu_eta should equal d(linkinv)/d(eta), verified numerically."""
        fam = BetaFamily(phi=3.0)
        eta = np.array([-2.0, 0.0, 2.0])
        h = 1e-7
        numerical = (fam.linkinv(eta + h) - fam.linkinv(eta - h)) / (2 * h)
        assert_allclose(fam.mu_eta(eta), numerical, rtol=1e-5)

    def test_dev_resids_perfect_fit(self):
        """Deviance residuals should be zero when y == mu."""
        fam = BetaFamily(phi=5.0)
        mu = np.array([0.2, 0.5, 0.8])
        wt = np.ones_like(mu)
        assert_allclose(fam.dev_resids(mu, mu, wt), 0.0, atol=1e-12)

    def test_dev_resids_finite(self):
        """Deviance residuals should be finite for interior y and mu.

        Note: Beta deviance using the mu=y "saturated" model can be negative
        because mu=y is *not* the true MLE (the sufficient statistic is
        (log y, log(1-y)), not y itself).  This is well-known in the
        beta regression literature (Ferrari & Cribari-Neto, 2004).
        """
        fam = BetaFamily(phi=3.0)
        rng = np.random.default_rng(42)
        y = rng.beta(2, 5, size=100)
        mu = rng.beta(2, 5, size=100)
        wt = np.ones(100)
        dr = fam.dev_resids(y, mu, wt)
        assert np.all(np.isfinite(dr))

    def test_dev_resids_known_value(self):
        """Beta deviance for y=0.3, mu=0.5, phi=2, wt=1.

        d_i = 2 * [log_beta_pdf(y; y, phi) - log_beta_pdf(y; mu, phi)]
        Using scipy as reference.
        """
        from scipy.stats import beta as beta_dist

        phi = 2.0
        fam = BetaFamily(phi=phi)
        y = np.array([0.3])
        mu = np.array([0.5])
        wt = np.array([1.0])
        # Saturated: a_sat=y*phi, b_sat=(1-y)*phi
        ll_sat = beta_dist.logpdf(0.3, a=0.3 * phi, b=0.7 * phi)
        ll_mod = beta_dist.logpdf(0.3, a=0.5 * phi, b=0.5 * phi)
        expected = 2.0 * (ll_sat - ll_mod)
        assert_allclose(fam.dev_resids(y, mu, wt), expected, atol=1e-10)

    def test_linkinv_extreme_eta(self):
        """Should not overflow for very large/small eta."""
        fam = BetaFamily()
        eta = np.array([-500.0, 500.0])
        mu = fam.linkinv(eta)
        assert np.all(np.isfinite(mu))
        assert mu[0] < 1e-10
        assert mu[1] > 1.0 - 1e-10


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


class TestWeights:
    """Deviance residuals should scale linearly with weights."""

    @pytest.mark.parametrize(
        "cls",
        [
            BinomialFamily,
            PoissonFamily,
            GaussianFamily,
            NegativeBinomial2Family,
            BetaFamily,
            ZeroInflatedNB2Family,
        ],
    )
    def test_dev_resids_weight_scaling(self, cls):
        fam = cls()
        if cls is BinomialFamily:
            y = np.array([1.0, 0.0, 1.0])
            mu = np.array([0.7, 0.3, 0.9])
        elif cls in (PoissonFamily, NegativeBinomial2Family, ZeroInflatedNB2Family):
            y = np.array([3.0, 0.0, 5.0])
            mu = np.array([2.0, 1.0, 4.0])
        elif cls is BetaFamily:
            y = np.array([0.2, 0.5, 0.8])
            mu = np.array([0.3, 0.6, 0.7])
        else:
            y = np.array([1.0, 2.0, 3.0])
            mu = np.array([1.5, 1.5, 1.5])

        wt1 = np.ones(3)
        wt2 = 3.0 * np.ones(3)
        assert_allclose(
            fam.dev_resids(y, mu, wt2),
            3.0 * fam.dev_resids(y, mu, wt1),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Helper: resolve_family
# ---------------------------------------------------------------------------


class TestResolveFamily:
    """Test the string → family resolver."""

    def test_string_binomial(self):
        from interlace.glmm_family import resolve_family

        fam = resolve_family("binomial")
        assert isinstance(fam, BinomialFamily)

    def test_string_poisson(self):
        from interlace.glmm_family import resolve_family

        fam = resolve_family("poisson")
        assert isinstance(fam, PoissonFamily)

    def test_string_gaussian(self):
        from interlace.glmm_family import resolve_family

        fam = resolve_family("gaussian")
        assert isinstance(fam, GaussianFamily)

    def test_string_negativebinomial(self):
        from interlace.glmm_family import resolve_family

        fam = resolve_family("negativebinomial")
        assert isinstance(fam, NegativeBinomial2Family)

    def test_passthrough_instance(self):
        from interlace.glmm_family import resolve_family

        fam = BinomialFamily()
        assert resolve_family(fam) is fam

    def test_string_beta(self):
        from interlace.glmm_family import resolve_family

        fam = resolve_family("beta")
        assert isinstance(fam, BetaFamily)

    def test_string_zeroinflated_negativebinomial(self):
        from interlace.glmm_family import resolve_family

        fam = resolve_family("zeroinflated_negativebinomial")
        assert isinstance(fam, ZeroInflatedNB2Family)

    def test_unknown_string_raises(self):
        from interlace.glmm_family import resolve_family

        with pytest.raises(ValueError, match="Unknown family"):
            resolve_family("gamma")
