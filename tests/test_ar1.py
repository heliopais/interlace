"""Tests for AR(1) residual correlation structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

import interlace
from interlace.correlation import AR1, _ar1_whiten_vector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ar1_dataset(
    rng: np.random.Generator,
    n_groups: int = 30,
    n_time: int = 10,
    rho_true: float = 0.7,
    beta0: float = 2.0,
    beta1: float = 0.5,
    sigma_b: float = 1.0,
    sigma_e: float = 1.0,
) -> pd.DataFrame:
    """Simulate longitudinal data with AR(1) residual correlation.

    Each group has `n_time` equally-spaced observations.
    Residuals within a group follow: e_t = rho * e_{t-1} + eps_t
    where eps_t ~ N(0, sigma_e^2 * (1 - rho^2)).
    """
    n = n_groups * n_time
    group = np.repeat(np.arange(n_groups), n_time)
    time = np.tile(np.arange(n_time), n_groups)
    x = rng.normal(size=n)

    # Random intercepts
    b = rng.normal(scale=sigma_b, size=n_groups)

    # AR(1) residuals within each group
    e = np.empty(n)
    innov_sd = sigma_e * np.sqrt(1.0 - rho_true**2)
    for g in range(n_groups):
        idx = slice(g * n_time, (g + 1) * n_time)
        e_g = np.empty(n_time)
        e_g[0] = rng.normal(scale=sigma_e)
        for t in range(1, n_time):
            e_g[t] = rho_true * e_g[t - 1] + rng.normal(scale=innov_sd)
        e[idx] = e_g

    y = beta0 + beta1 * x + b[group] + e

    return pd.DataFrame({"y": y, "x": x, "group": group, "time": time})


def _make_unequal_time_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate AR(1) data with unequally spaced time points."""
    n_groups = 20
    rho_true = 0.6
    sigma_b = 1.0
    sigma_e = 1.0

    rows = []
    for g in range(n_groups):
        # Random number of time points (5-12) at irregular intervals
        n_t = rng.integers(5, 13)
        times = np.sort(rng.uniform(0, 10, size=n_t))
        x = rng.normal(size=n_t)
        b_g = rng.normal(scale=sigma_b)

        # AR(1) residuals with unequal spacing
        e = np.empty(n_t)
        e[0] = rng.normal(scale=sigma_e)
        for t in range(1, n_t):
            dt = times[t] - times[t - 1]
            rho_dt = rho_true**dt
            e[t] = rho_dt * e[t - 1] + rng.normal(
                scale=sigma_e * np.sqrt(1.0 - rho_dt**2)
            )

        y = 1.0 + 0.3 * x + b_g + e
        for t in range(n_t):
            rows.append({"y": y[t], "x": x[t], "group": g, "time": times[t]})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Whitening transform tests
# ---------------------------------------------------------------------------


class TestAR1Whitening:
    """Test the AR(1) whitening transform correctness."""

    def test_whiten_vector_equally_spaced(self):
        """Whitening should produce uncorrelated residuals from AR(1) input."""
        rho = 0.7
        n = 100
        v = np.arange(1.0, n + 1.0)

        result = _ar1_whiten_vector(v, rho, dt=np.ones(n - 1))

        # First element unchanged
        assert result[0] == v[0]
        # Subsequent: (v_i - rho * v_{i-1}) / sqrt(1 - rho^2)
        expected_1 = (v[1] - rho * v[0]) / np.sqrt(1.0 - rho**2)
        np.testing.assert_allclose(result[1], expected_1, rtol=1e-12)

    def test_whiten_vector_unequal_spacing(self):
        """Whitening with unequal time gaps uses rho^dt."""
        rho = 0.5
        v = np.array([1.0, 2.0, 4.0])
        dt = np.array([1.0, 3.0])  # gaps between consecutive observations

        result = _ar1_whiten_vector(v, rho, dt)

        # First element unchanged
        assert result[0] == 1.0
        # Second: (2 - 0.5^1 * 1) / sqrt(1 - 0.5^2) = 1.5 / sqrt(0.75)
        expected_1 = (2.0 - 0.5 * 1.0) / np.sqrt(1.0 - 0.25)
        np.testing.assert_allclose(result[1], expected_1, rtol=1e-12)
        # Third: (4 - 0.5^3 * 2) / sqrt(1 - 0.5^6)
        rho_dt = 0.5**3
        expected_2 = (4.0 - rho_dt * 2.0) / np.sqrt(1.0 - rho_dt**2)
        np.testing.assert_allclose(result[2], expected_2, rtol=1e-12)

    def test_whiten_rho_zero_is_identity(self):
        """rho=0 means iid residuals; whitening should be identity."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        dt = np.ones(3)

        result = _ar1_whiten_vector(v, rho=0.0, dt=dt)

        np.testing.assert_allclose(result, v, rtol=1e-12)

    def test_whiten_produces_uncorrelated_residuals(self):
        """Whitening AR(1) draws should produce approximately iid residuals."""
        rng = np.random.default_rng(123)
        rho = 0.8
        n = 5000
        sigma = 1.0

        # Generate AR(1) process
        e = np.empty(n)
        e[0] = rng.normal(scale=sigma)
        innov_sd = sigma * np.sqrt(1.0 - rho**2)
        for t in range(1, n):
            e[t] = rho * e[t - 1] + rng.normal(scale=innov_sd)

        # Whiten
        whitened = _ar1_whiten_vector(e, rho, dt=np.ones(n - 1))

        # Lag-1 autocorrelation of whitened should be near zero
        acf1 = np.corrcoef(whitened[:-1], whitened[1:])[0, 1]
        assert abs(acf1) < 0.05, f"lag-1 autocorrelation after whitening: {acf1:.3f}"


# ---------------------------------------------------------------------------
# AR1 class tests
# ---------------------------------------------------------------------------


class TestAR1Class:
    """Test the AR1 correlation structure class."""

    def test_constructor(self):
        """AR1 takes a time column name."""
        cor = AR1("time")
        assert cor.time_col == "time"
        assert cor.n_corr_params == 1

    def test_log_det_R_zero_rho(self):
        """log|R| should be 0 when rho=0 (R=I)."""
        cor = AR1("time")
        groups = np.array([0, 0, 0, 1, 1, 1])
        times = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        cor.setup(groups, times)

        log_det = cor.log_det_R(np.array([0.0]))
        np.testing.assert_allclose(log_det, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Integration: fit() with AR(1) correlation
# ---------------------------------------------------------------------------


class TestFitAR1:
    """Test that fit() accepts and uses AR(1) correlation."""

    def test_fit_accepts_correlation(self):
        """fit() should accept a correlation=AR1(...) parameter."""
        rng = np.random.default_rng(42)
        df = _make_ar1_dataset(rng)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=AR1("time"),
        )

        assert result.converged
        assert hasattr(result, "fe_params")

    def test_ar1_recovers_rho(self):
        """Estimated rho should be close to the true rho."""
        rng = np.random.default_rng(42)
        rho_true = 0.7
        df = _make_ar1_dataset(rng, rho_true=rho_true, n_groups=50, n_time=15)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=AR1("time"),
        )

        assert result.converged
        # rho should be stored on the result
        rho_hat = result.correlation_params["rho"]
        assert abs(rho_hat - rho_true) < 0.15, (
            f"rho_hat={rho_hat:.3f}, expected ~{rho_true}"
        )

    def test_ar1_recovers_fixed_effects(self):
        """Fixed effects should be close to true values under AR(1)."""
        rng = np.random.default_rng(42)
        df = _make_ar1_dataset(
            rng, beta0=2.0, beta1=0.5, rho_true=0.6, n_groups=50, n_time=15
        )

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=AR1("time"),
        )

        assert result.converged
        np.testing.assert_allclose(result.fe_params["Intercept"], 2.0, atol=0.5)
        np.testing.assert_allclose(result.fe_params["x"], 0.5, atol=0.2)

    def test_ar1_improves_over_iid(self):
        """AR(1) model should have better (lower) AIC than iid on AR(1) data."""
        rng = np.random.default_rng(42)
        df = _make_ar1_dataset(rng, rho_true=0.7, n_groups=50, n_time=15)

        result_iid = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
        )
        result_ar1 = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=AR1("time"),
        )

        assert result_ar1.aic < result_iid.aic, (
            f"AR1 AIC={result_ar1.aic:.1f} should be < iid AIC={result_iid.aic:.1f}"
        )

    def test_ar1_on_iid_data_rho_near_zero(self):
        """On iid data, the estimated rho should be near zero."""
        rng = np.random.default_rng(42)
        df = _make_ar1_dataset(rng, rho_true=0.0, n_groups=30, n_time=10)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=AR1("time"),
        )

        assert result.converged
        rho_hat = result.correlation_params["rho"]
        assert abs(rho_hat) < 0.15, f"rho_hat={rho_hat:.3f}, expected ~0.0"

    def test_ar1_with_unequal_spacing(self):
        """AR(1) should work with unequally spaced time points."""
        rng = np.random.default_rng(42)
        df = _make_unequal_time_dataset(rng)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=AR1("time"),
        )

        assert result.converged
        assert "rho" in result.correlation_params

    def test_ar1_with_ml(self):
        """AR(1) should work with method='ML'."""
        rng = np.random.default_rng(42)
        df = _make_ar1_dataset(rng)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            method="ML",
            correlation=AR1("time"),
        )

        assert result.converged

    def test_ar1_negative_rho(self):
        """AR(1) should handle negative autocorrelation."""
        rng = np.random.default_rng(42)
        df = _make_ar1_dataset(rng, rho_true=-0.5, n_groups=50, n_time=15)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=AR1("time"),
        )

        assert result.converged
        rho_hat = result.correlation_params["rho"]
        assert rho_hat < 0, f"rho_hat={rho_hat:.3f}, expected negative"
