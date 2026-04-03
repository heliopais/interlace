"""Tests for GLMMFamily protocol and concrete family implementations."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from interlace.glmm_family import (
    BinomialFamily,
    GaussianFamily,
    GLMMFamily,
    PoissonFamily,
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

    @pytest.mark.parametrize("cls", [BinomialFamily, PoissonFamily, GaussianFamily])
    def test_is_runtime_checkable(self, cls):
        family = cls()
        assert isinstance(family, GLMMFamily)

    @pytest.mark.parametrize("cls", [BinomialFamily, PoissonFamily, GaussianFamily])
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
# Weights
# ---------------------------------------------------------------------------


class TestWeights:
    """Deviance residuals should scale linearly with weights."""

    @pytest.mark.parametrize("cls", [BinomialFamily, PoissonFamily, GaussianFamily])
    def test_dev_resids_weight_scaling(self, cls):
        fam = cls()
        if cls is BinomialFamily:
            y = np.array([1.0, 0.0, 1.0])
            mu = np.array([0.7, 0.3, 0.9])
        elif cls is PoissonFamily:
            y = np.array([3.0, 0.0, 5.0])
            mu = np.array([2.0, 1.0, 4.0])
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

    def test_passthrough_instance(self):
        from interlace.glmm_family import resolve_family

        fam = BinomialFamily()
        assert resolve_family(fam) is fam

    def test_unknown_string_raises(self):
        from interlace.glmm_family import resolve_family

        with pytest.raises(ValueError, match="Unknown family"):
            resolve_family("gamma")
