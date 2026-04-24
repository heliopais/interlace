"""Tests for compound symmetry (exchangeable) residual correlation structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

import interlace
from interlace.correlation import CompoundSymmetry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cs_dataset(
    rng: np.random.Generator,
    n_groups: int = 30,
    n_time: int = 8,
    rho_true: float = 0.5,
    beta0: float = 2.0,
    beta1: float = 0.5,
    sigma_b: float = 1.0,
    sigma_e: float = 1.0,
) -> pd.DataFrame:
    """Simulate longitudinal data with compound symmetry residual correlation.

    Within each group the residual vector has covariance
    sigma_e^2 * [(1-rho)*I + rho*J], i.e. all pairs share correlation rho.
    """
    n = n_groups * n_time
    group = np.repeat(np.arange(n_groups), n_time)
    time = np.tile(np.arange(n_time), n_groups)
    x = rng.normal(size=n)

    # Random intercepts
    b = rng.normal(scale=sigma_b, size=n_groups)

    # Compound symmetry residuals: e = sqrt(rho)*u_g + sqrt(1-rho)*eps_i
    # where u_g ~ N(0, sigma_e^2) shared within group, eps_i ~ N(0, sigma_e^2) iid
    e = np.empty(n)
    for g in range(n_groups):
        idx = slice(g * n_time, (g + 1) * n_time)
        u_g = rng.normal() * sigma_e * np.sqrt(rho_true)
        eps = rng.normal(size=n_time) * sigma_e * np.sqrt(1.0 - rho_true)
        e[idx] = u_g + eps

    y = beta0 + beta1 * x + b[group] + e

    return pd.DataFrame({"y": y, "x": x, "group": group, "time": time})


# ---------------------------------------------------------------------------
# CompoundSymmetry class tests
# ---------------------------------------------------------------------------


class TestCompoundSymmetryClass:
    """Test the CompoundSymmetry correlation structure class."""

    def test_constructor(self):
        """CS takes a group column name."""
        cor = CompoundSymmetry("group")
        assert cor.time_col == "group"
        assert cor.n_corr_params == 1

    def test_log_det_R_zero_rho(self):
        """log|R| should be 0 when rho=0 (R=I)."""
        cor = CompoundSymmetry("group")
        groups = np.array([0, 0, 0, 1, 1, 1])
        times = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        cor.setup(groups, times)

        log_det = cor.log_det_R(np.array([0.0]))
        np.testing.assert_allclose(log_det, 0.0, atol=1e-14)

    def test_log_det_R_known_value(self):
        """log|R| for CS with rho=0.5, group size 3:
        det(R) = (1 - rho)^{n-1} * (1 + (n-1)*rho)
               = 0.5^2 * 2.0 = 0.5
        log|R| = log(0.5) = -0.693...
        """
        cor = CompoundSymmetry("group")
        groups = np.array([0, 0, 0])
        times = np.array([0.0, 1.0, 2.0])
        cor.setup(groups, times)

        log_det = cor.log_det_R(np.array([0.5]))
        expected = np.log(0.5**2 * (1 + 2 * 0.5))  # log(0.5)
        np.testing.assert_allclose(log_det, expected, rtol=1e-12)

    def test_whiten_recovers_identity_covariance(self):
        """Whitening CS-correlated data should yield approximately iid residuals.

        Stack whitened group vectors into an (n_groups x n_t) matrix and check
        that the sample covariance is close to sigma^2 * I.
        """
        rng = np.random.default_rng(42)
        rho = 0.6
        n_g = 500
        n_t = 5
        sigma = 1.0

        # Generate CS-correlated residuals
        e_all = []
        for _ in range(n_g):
            u = rng.normal() * sigma * np.sqrt(rho)
            eps = rng.normal(size=n_t) * sigma * np.sqrt(1 - rho)
            e_all.append(u + eps)
        e = np.concatenate(e_all)

        groups = np.repeat(np.arange(n_g), n_t)
        times = np.tile(np.arange(n_t, dtype=float), n_g)

        cor = CompoundSymmetry("group")
        cor.setup(groups, times)

        import scipy.sparse as sp

        y = e
        X = np.ones((len(e), 1))
        Z = sp.csc_matrix((len(e), 1))  # dummy

        y_w, _, _ = cor.whiten_data(y, X, Z, np.array([rho]))

        # Stack into (n_g x n_t) and compute sample correlation matrix
        W = y_w.reshape(n_g, n_t)
        sample_corr = np.corrcoef(W.T)  # (n_t x n_t)

        # Off-diagonal elements should be near zero
        off_diag = sample_corr[np.triu_indices(n_t, k=1)]
        assert np.max(np.abs(off_diag)) < 0.15, (
            f"max off-diagonal corr = {np.max(np.abs(off_diag)):.3f}"
        )


# ---------------------------------------------------------------------------
# Integration: fit() with compound symmetry
# ---------------------------------------------------------------------------


class TestFitCS:
    """Test that fit() accepts and uses compound symmetry correlation."""

    def test_fit_accepts_cs(self):
        """fit() should accept correlation=CompoundSymmetry(...)."""
        rng = np.random.default_rng(42)
        df = _make_cs_dataset(rng)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=CompoundSymmetry("group"),
        )

        assert result.converged
        assert hasattr(result, "fe_params")

    def test_cs_recovers_rho(self):
        """Estimated rho should be close to the true rho."""
        rng = np.random.default_rng(42)
        rho_true = 0.5
        df = _make_cs_dataset(rng, rho_true=rho_true, n_groups=50, n_time=10)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=CompoundSymmetry("group"),
        )

        assert result.converged
        rho_hat = result.correlation_params["rho"]
        assert abs(rho_hat - rho_true) < 0.15, (
            f"rho_hat={rho_hat:.3f}, expected ~{rho_true}"
        )

    def test_cs_recovers_fixed_effects(self):
        """Fixed effects should be close to true values under CS."""
        rng = np.random.default_rng(42)
        df = _make_cs_dataset(
            rng, beta0=2.0, beta1=0.5, rho_true=0.4, n_groups=50, n_time=10
        )

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=CompoundSymmetry("group"),
        )

        assert result.converged
        np.testing.assert_allclose(result.fe_params["Intercept"], 2.0, atol=0.5)
        np.testing.assert_allclose(result.fe_params["x"], 0.5, atol=0.2)

    def test_cs_aic_close_to_iid(self):
        """CS + random intercept is confounded (both add constant within-group
        correlation), so AIC should be within ~3 of the iid model (penalty for
        the redundant rho parameter)."""
        rng = np.random.default_rng(42)
        df = _make_cs_dataset(rng, rho_true=0.5, n_groups=50, n_time=10)

        result_iid = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
        )
        result_cs = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=CompoundSymmetry("group"),
        )

        # Log-likelihoods should be nearly equal; AIC difference is just the
        # penalty for the extra parameter (~2).
        assert abs(result_cs.aic - result_iid.aic) < 5.0, (
            f"CS AIC={result_cs.aic:.1f} vs iid AIC={result_iid.aic:.1f}"
        )

    def test_cs_on_iid_data_rho_near_zero(self):
        """On iid data, the estimated rho should be near zero.

        Note: CS rho is partially confounded with the random intercept
        (both add constant within-group correlation), so the tolerance is wider.
        """
        rng = np.random.default_rng(42)
        df = _make_cs_dataset(rng, rho_true=0.0, n_groups=50, n_time=10)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=CompoundSymmetry("group"),
        )

        assert result.converged
        rho_hat = result.correlation_params["rho"]
        assert abs(rho_hat) < 0.30, f"rho_hat={rho_hat:.3f}, expected ~0.0"

    def test_cs_with_ml(self):
        """CS should work with method='ML'."""
        rng = np.random.default_rng(42)
        df = _make_cs_dataset(rng)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            method="ML",
            correlation=CompoundSymmetry("group"),
        )

        assert result.converged

    def test_cs_unequal_group_sizes(self):
        """CS should handle groups of different sizes."""
        rng = np.random.default_rng(42)
        rows = []
        rho_true = 0.4
        sigma_e = 1.0
        for g in range(30):
            n_t = rng.integers(4, 12)
            x = rng.normal(size=n_t)
            b_g = rng.normal()
            u = rng.normal() * sigma_e * np.sqrt(rho_true)
            eps = rng.normal(size=n_t) * sigma_e * np.sqrt(1 - rho_true)
            y = 1.0 + 0.3 * x + b_g + u + eps
            for t in range(n_t):
                rows.append({"y": y[t], "x": x[t], "group": g, "time": t})

        df = pd.DataFrame(rows)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            correlation=CompoundSymmetry("group"),
        )

        assert result.converged
        assert "rho" in result.correlation_params
