"""Tests for interlace.ols_influence_measures — single-QR OLS influence diagnostics.

Acceptance criteria:
  - Returns dict with keys: cooks, hat, dfbetas, dffits, residuals, sigma
  - hat sums to p (trace of hat matrix)
  - cooks, hat are non-negative
  - dffits, residuals are finite
  - sigma is a positive scalar
  - hat matches QR-based diagonal (‖Qᵢ‖²)
  - cooks matches closed-form formula given hat
  - dffits matches closed-form formula given hat and LOO sigma
  - dfbetas matches ols_dfbetas_qr exactly (same QR path)
  - Numerically agrees with brute-force LOO-refit Cook's D
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_ols():
    """Small OLS: n=40, p=3 (intercept + 2 predictors)."""
    rng = np.random.default_rng(42)
    n = 40
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 + 0.5 * x1 - 0.8 * x2 + rng.normal(0, 0.5, n)
    X = sm.add_constant(np.column_stack([x1, x2]))
    return sm.OLS(y, X).fit()


@pytest.fixture(scope="module")
def wide_ols():
    """Wide OLS: n=200, p=40 (intercept + 39 predictors)."""
    rng = np.random.default_rng(7)
    n, p = 200, 40
    X = sm.add_constant(rng.standard_normal((n, p - 1)))
    y = X @ rng.standard_normal(p) + rng.normal(0, 1.0, n)
    return sm.OLS(y, X).fit()


@pytest.fixture(scope="module")
def intercept_only():
    """Minimal model: intercept only, n=20."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20)
    X = np.ones((20, 1))
    return sm.OLS(y, X).fit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brute_force_cooks(model) -> np.ndarray:
    """Exact LOO-refit Cook's D for reference (slow, small n only)."""
    X = model.model.exog
    y = model.model.endog
    n, p = X.shape
    beta_full = np.asarray(model.params)
    XtXinv = np.asarray(model.normalized_cov_params)

    cooks = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        Xi, yi = X[mask], y[mask]
        beta_i = np.asarray(sm.OLS(yi, Xi).fit().params)
        db = beta_full - beta_i
        cooks[i] = (db @ np.linalg.inv(XtXinv * model.mse_resid) @ db) / p

    return cooks


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestOlsInfluenceMeasuresStructure:
    def test_returns_dict(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        assert isinstance(out, dict)

    def test_required_keys_present(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        assert set(out.keys()) == {
            "cooks",
            "hat",
            "dfbetas",
            "dffits",
            "residuals",
            "sigma",
        }

    def test_array_shapes(self, simple_ols):
        from interlace import ols_influence_measures

        n, p = simple_ols.model.exog.shape
        out = ols_influence_measures(simple_ols)
        assert out["cooks"].shape == (n,)
        assert out["hat"].shape == (n,)
        assert out["dffits"].shape == (n,)
        assert out["residuals"].shape == (n,)
        assert out["dfbetas"].shape == (n, p)

    def test_sigma_is_scalar(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        assert np.ndim(out["sigma"]) == 0 or out["sigma"].shape == ()

    def test_all_arrays_finite(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        for key in ("cooks", "hat", "dfbetas", "dffits", "residuals"):
            assert np.all(np.isfinite(out[key])), f"{key} contains non-finite values"

    def test_wide_model_shape(self, wide_ols):
        from interlace import ols_influence_measures

        n, p = wide_ols.model.exog.shape
        out = ols_influence_measures(wide_ols)
        assert out["dfbetas"].shape == (n, p)
        assert out["hat"].shape == (n,)

    def test_intercept_only_model(self, intercept_only):
        from interlace import ols_influence_measures

        n, p = intercept_only.model.exog.shape
        out = ols_influence_measures(intercept_only)
        assert out["dfbetas"].shape == (n, p)
        assert np.all(np.isfinite(out["cooks"]))


# ---------------------------------------------------------------------------
# Non-negativity / bounds
# ---------------------------------------------------------------------------


class TestOlsInfluenceMeasuresBounds:
    def test_hat_non_negative(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        assert np.all(out["hat"] >= 0.0)

    def test_hat_at_most_one(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        assert np.all(out["hat"] <= 1.0 + 1e-9)

    def test_cooks_non_negative(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        assert np.all(out["cooks"] >= 0.0)

    def test_sigma_positive(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        assert float(out["sigma"]) > 0.0

    def test_hat_trace_equals_p(self, simple_ols):
        """Trace of hat matrix equals rank p (fundamental OLS identity)."""
        from interlace import ols_influence_measures

        p = simple_ols.model.exog.shape[1]
        out = ols_influence_measures(simple_ols)
        np.testing.assert_allclose(out["hat"].sum(), p, rtol=1e-9)

    def test_hat_trace_equals_p_wide(self, wide_ols):
        from interlace import ols_influence_measures

        p = wide_ols.model.exog.shape[1]
        out = ols_influence_measures(wide_ols)
        np.testing.assert_allclose(out["hat"].sum(), p, rtol=1e-9)


# ---------------------------------------------------------------------------
# Correctness: internal consistency
# ---------------------------------------------------------------------------


class TestOlsInfluenceMeasuresCorrectness:
    def test_residuals_match_model(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        np.testing.assert_allclose(
            out["residuals"], np.asarray(simple_ols.resid), atol=1e-12
        )

    def test_sigma_matches_model_mse(self, simple_ols):
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        np.testing.assert_allclose(
            float(out["sigma"]), np.sqrt(simple_ols.mse_resid), rtol=1e-10
        )

    def test_cooks_formula_consistency(self, simple_ols):
        """Cook's D = e²·h / (p·MSE·(1-h)²) — verify from returned hat and residuals."""
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        e = out["residuals"]
        h = out["hat"]
        p = simple_ols.model.exog.shape[1]
        mse = simple_ols.mse_resid
        expected = e**2 * h / (p * mse * np.maximum(1 - h, 1e-10) ** 2)
        np.testing.assert_allclose(out["cooks"], expected, rtol=1e-9)

    def test_dfbetas_matches_ols_dfbetas_qr(self, simple_ols):
        """dfbetas must be identical to ols_dfbetas_qr (same algorithm, same QR)."""
        from interlace import ols_dfbetas_qr, ols_influence_measures

        out = ols_influence_measures(simple_ols)
        ref = ols_dfbetas_qr(simple_ols)
        np.testing.assert_allclose(out["dfbetas"], ref, atol=1e-12)

    def test_dfbetas_matches_ols_dfbetas_qr_wide(self, wide_ols):
        from interlace import ols_dfbetas_qr, ols_influence_measures

        out = ols_influence_measures(wide_ols)
        ref = ols_dfbetas_qr(wide_ols)
        np.testing.assert_allclose(out["dfbetas"], ref, atol=1e-12)

    def test_hat_matches_qr_diagonal(self, simple_ols):
        """Hat diagonal must equal ‖Qᵢ‖² from thin QR."""
        from interlace import ols_influence_measures

        X = np.asarray(simple_ols.model.exog)
        Q, _ = np.linalg.qr(X, mode="reduced")
        expected_hat = np.einsum("ij,ij->i", Q, Q)

        out = ols_influence_measures(simple_ols)
        np.testing.assert_allclose(out["hat"], expected_hat, atol=1e-12)

    def test_cooks_matches_brute_force_loo(self, simple_ols):
        """Cook's D must match brute-force LOO-refit reference."""
        from interlace import ols_influence_measures

        out = ols_influence_measures(simple_ols)
        ref = _brute_force_cooks(simple_ols)
        np.testing.assert_allclose(out["cooks"], ref, rtol=1e-5)


# ---------------------------------------------------------------------------
# Influential observation detection
# ---------------------------------------------------------------------------


class TestOlsInfluenceMeasuresInfluential:
    def test_outlier_has_large_cooks(self):
        """Injected outlier should produce the largest Cook's D."""
        from interlace import ols_influence_measures

        rng = np.random.default_rng(99)
        n = 50
        X = sm.add_constant(rng.standard_normal((n, 2)))
        y = X @ np.array([1.0, 0.5, -0.5]) + rng.normal(0, 0.3, n)

        # Inject high-leverage outlier at index 0
        X[0, 1] = 15.0
        y[0] = -20.0
        model = sm.OLS(y, X).fit()

        out = ols_influence_measures(model)
        assert np.argmax(out["cooks"]) == 0, (
            "Outlier at index 0 should have max Cook's D"
        )

    def test_outlier_has_large_dfbetas(self):
        """Injected outlier should have the largest max-abs DFBETAS row."""
        from interlace import ols_influence_measures

        rng = np.random.default_rng(88)
        n = 50
        X = sm.add_constant(rng.standard_normal((n, 2)))
        y = X @ np.array([1.0, 0.5, -0.5]) + rng.normal(0, 0.3, n)
        X[0, 1] = 15.0
        y[0] = -20.0
        model = sm.OLS(y, X).fit()

        out = ols_influence_measures(model)
        max_abs_row = np.max(np.abs(out["dfbetas"]), axis=1)
        assert np.argmax(max_abs_row) == 0
