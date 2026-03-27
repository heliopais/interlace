"""Tests for quantreg_ker_se: Hall-Sheather kernel SE matching R's quantreg.

Acceptance criteria (GitHub issue #9):
  - _hall_sheather_bandwidth replicates R's bandwidth.rq(tau, n, hs=TRUE)
  - _bofinger_bandwidth replicates R's bandwidth.rq(tau, n, hs=FALSE)
  - quantreg_ker_se(residuals, X, tau, hs=True) returns SE array of length p
  - SEs are positive
  - Hall-Sheather and Bofinger bandwidths differ → different SEs
  - Formula: f_hat = 2h / (Q(tau+h) - Q(tau-h)); cov = tau*(1-tau)/f_hat^2 * inv(X'X)
  - Invalid bandwidth (tau+h>1 or tau-h<0) raises ValueError
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from interlace.quantreg import (
    _bofinger_bandwidth,
    _hall_sheather_bandwidth,
    quantreg_ker_se,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_qr_data():
    """Synthetic y ~ intercept + male data; residuals from median QR fit."""
    import statsmodels.regression.quantile_regression as sqr

    rng = np.random.default_rng(42)
    n = 400
    male = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    y = 10_000 + 3_000 * male + rng.normal(0, 2_000, n)
    X = np.column_stack([np.ones(n), male])
    result = sqr.QuantReg(y, X).fit(q=0.5, disp=False)
    return {"residuals": result.resid, "X": X, "n": n}


# ---------------------------------------------------------------------------
# Hall-Sheather bandwidth
# ---------------------------------------------------------------------------


class TestHallSheatherBandwidth:
    def test_tau_half_n800(self):
        """Matches manual computation of R's bandwidth.rq(0.5, 800, hs=TRUE)."""
        n, tau = 800, 0.5
        x = scipy_stats.norm.ppf(tau)
        f = scipy_stats.norm.pdf(x)
        z = scipy_stats.norm.ppf(0.975)
        expected = (
            n ** (-1 / 3) * z ** (2 / 3) * ((1.5 * f**2) / (2 * x**2 + 1)) ** (1 / 3)
        )
        assert abs(_hall_sheather_bandwidth(n, tau) - expected) < 1e-12

    def test_tau_three_quarters_n500(self):
        """Matches manual computation for tau=0.75."""
        n, tau = 500, 0.75
        x = scipy_stats.norm.ppf(tau)
        f = scipy_stats.norm.pdf(x)
        z = scipy_stats.norm.ppf(0.975)
        expected = (
            n ** (-1 / 3) * z ** (2 / 3) * ((1.5 * f**2) / (2 * x**2 + 1)) ** (1 / 3)
        )
        assert abs(_hall_sheather_bandwidth(n, tau) - expected) < 1e-12

    def test_custom_alpha(self):
        """alpha parameter is passed through correctly."""
        n, tau, alpha = 300, 0.5, 0.01
        x = scipy_stats.norm.ppf(tau)
        f = scipy_stats.norm.pdf(x)
        z = scipy_stats.norm.ppf(1 - alpha / 2)
        expected = (
            n ** (-1 / 3) * z ** (2 / 3) * ((1.5 * f**2) / (2 * x**2 + 1)) ** (1 / 3)
        )
        assert abs(_hall_sheather_bandwidth(n, tau, alpha=alpha) - expected) < 1e-12

    def test_decreases_with_n(self):
        """Larger n → narrower bandwidth."""
        assert _hall_sheather_bandwidth(1000, 0.5) < _hall_sheather_bandwidth(100, 0.5)

    def test_positive(self):
        assert _hall_sheather_bandwidth(200, 0.5) > 0


# ---------------------------------------------------------------------------
# Bofinger bandwidth
# ---------------------------------------------------------------------------


class TestBofingerBandwidth:
    def test_tau_half_n800(self):
        """Matches manual computation of R's bandwidth.rq(0.5, 800, hs=FALSE)."""
        n, tau = 800, 0.5
        x = scipy_stats.norm.ppf(tau)
        f = scipy_stats.norm.pdf(x)
        expected = ((4.5 * f**4) / (2 * x**2 + 1) ** 2) ** 0.2 * n ** (-0.2)
        assert abs(_bofinger_bandwidth(n, tau) - expected) < 1e-12

    def test_differs_from_hall_sheather(self):
        """Two bandwidth rules should not give the same result."""
        assert _bofinger_bandwidth(800, 0.5) != pytest.approx(
            _hall_sheather_bandwidth(800, 0.5)
        )

    def test_positive(self):
        assert _bofinger_bandwidth(200, 0.5) > 0


# ---------------------------------------------------------------------------
# quantreg_ker_se
# ---------------------------------------------------------------------------


class TestQuantregKerSE:
    def test_returns_array_length_p(self, simple_qr_data):
        se = quantreg_ker_se(simple_qr_data["residuals"], simple_qr_data["X"], tau=0.5)
        assert se.shape == (2,)

    def test_all_positive(self, simple_qr_data):
        se = quantreg_ker_se(simple_qr_data["residuals"], simple_qr_data["X"], tau=0.5)
        assert np.all(se > 0)

    def test_matches_explicit_formula_hs_true(self, simple_qr_data):
        """SE matches step-by-step replication of R's formula (hs=TRUE)."""
        residuals = simple_qr_data["residuals"]
        X = simple_qr_data["X"]
        n = simple_qr_data["n"]
        tau = 0.5

        # Bandwidth
        x = scipy_stats.norm.ppf(tau)
        f = scipy_stats.norm.pdf(x)
        z = scipy_stats.norm.ppf(0.975)
        h = n ** (-1 / 3) * z ** (2 / 3) * ((1.5 * f**2) / (2 * x**2 + 1)) ** (1 / 3)

        # Sparsity
        bhi = np.quantile(residuals, tau + h)
        blo = np.quantile(residuals, tau - h)
        f_hat = 2 * h / (bhi - blo)

        # Covariance
        XtX_inv = np.linalg.inv(X.T @ X)
        cov = (tau * (1 - tau) / f_hat**2) * XtX_inv
        expected = np.sqrt(np.diag(cov))

        np.testing.assert_allclose(
            quantreg_ker_se(residuals, X, tau=0.5, hs=True), expected, rtol=1e-10
        )

    def test_matches_explicit_formula_hs_false(self, simple_qr_data):
        """SE matches step-by-step replication of R's formula (hs=FALSE / Bofinger)."""
        residuals = simple_qr_data["residuals"]
        X = simple_qr_data["X"]
        n = simple_qr_data["n"]
        tau = 0.5

        # Bofinger bandwidth
        x = scipy_stats.norm.ppf(tau)
        f = scipy_stats.norm.pdf(x)
        h = ((4.5 * f**4) / (2 * x**2 + 1) ** 2) ** 0.2 * n ** (-0.2)

        bhi = np.quantile(residuals, tau + h)
        blo = np.quantile(residuals, tau - h)
        f_hat = 2 * h / (bhi - blo)

        XtX_inv = np.linalg.inv(X.T @ X)
        cov = (tau * (1 - tau) / f_hat**2) * XtX_inv
        expected = np.sqrt(np.diag(cov))

        np.testing.assert_allclose(
            quantreg_ker_se(residuals, X, tau=0.5, hs=False), expected, rtol=1e-10
        )

    def test_hs_true_and_false_differ(self, simple_qr_data):
        """Different bandwidth rules produce different SEs."""
        residuals = simple_qr_data["residuals"]
        X = simple_qr_data["X"]
        se_hs = quantreg_ker_se(residuals, X, tau=0.5, hs=True)
        se_bof = quantreg_ker_se(residuals, X, tau=0.5, hs=False)
        assert not np.allclose(se_hs, se_bof)

    def test_invalid_bandwidth_raises(self):
        """tau+h > 1 should raise ValueError."""
        rng = np.random.default_rng(0)
        # Very small n → large bandwidth; extreme tau → tau+h > 1
        residuals = rng.standard_normal(5)
        X = np.column_stack([np.ones(5), np.arange(5)])
        with pytest.raises(ValueError, match="bandwidth"):
            quantreg_ker_se(residuals, X, tau=0.99, hs=True)
