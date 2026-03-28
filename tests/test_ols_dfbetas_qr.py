"""Tests for interlace.ols_dfbetas_qr — vectorised QR-based DFBETAS for OLS.

Acceptance criteria:
  - Output shape is (n, p) matching design matrix dimensions
  - All values are finite
  - Numerically matches brute-force LOO-refit DFBETAS (R convention: LOO sigma)
  - Works correctly on wide design matrices (many dummy columns)
  - No Python loops (behaviour tested, not code structure — via timing/shape)
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm


@pytest.fixture(scope="module")
def simple_ols():
    """Small OLS dataset: n=30, p=3 (intercept + 2 predictors)."""
    rng = np.random.default_rng(42)
    n = 30
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 + 0.5 * x1 - 0.8 * x2 + rng.normal(0, 0.5, n)
    X = sm.add_constant(np.column_stack([x1, x2]))
    model = sm.OLS(y, X).fit()
    return model


@pytest.fixture(scope="module")
def wide_ols():
    """Wide OLS: n=200, p=40 (intercept + 39 predictors / dummies)."""
    rng = np.random.default_rng(7)
    n, p = 200, 40
    X_raw = rng.standard_normal((n, p - 1))
    X = sm.add_constant(X_raw)
    y = X @ rng.standard_normal(p) + rng.normal(0, 1.0, n)
    return sm.OLS(y, X).fit()


def _brute_force_dfbetas(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> np.ndarray:
    """Exact LOO-refit DFBETAS for validation (slow, only for small n).

    Formula: DFBETAS[i,j] = (β̂_j − β̂_{j(i)}) / (s_(i) · sqrt(c_jj))
    where c_jj = [(X'X)⁻¹]_jj and s_(i) is the LOO sigma.
    Matches R's influence.measures() convention.
    """
    X = model.model.exog
    y = model.model.endog
    n, p = X.shape
    beta_full = np.asarray(model.params)
    # sqrt of diagonal of (X'X)^{-1} — no sigma factor (R convention)
    se_coef = np.sqrt(np.diag(np.asarray(model.normalized_cov_params)))

    dfb = np.empty((n, p))
    for i in range(n):
        mask = np.arange(n) != i
        Xi, yi = X[mask], y[mask]
        res_i = sm.OLS(yi, Xi).fit()
        beta_i = np.asarray(res_i.params)
        s_i = np.sqrt(res_i.mse_resid)
        dfb[i] = (beta_full - beta_i) / (s_i * se_coef)

    return dfb


class TestOlsDfbetasQrShape:
    def test_shape_small(self, simple_ols):
        from interlace import ols_dfbetas_qr

        dfb = ols_dfbetas_qr(simple_ols)
        n, p = simple_ols.model.exog.shape
        assert dfb.shape == (n, p)

    def test_shape_wide(self, wide_ols):
        from interlace import ols_dfbetas_qr

        dfb = ols_dfbetas_qr(wide_ols)
        n, p = wide_ols.model.exog.shape
        assert dfb.shape == (n, p)

    def test_returns_numpy_array(self, simple_ols):
        from interlace import ols_dfbetas_qr

        dfb = ols_dfbetas_qr(simple_ols)
        assert isinstance(dfb, np.ndarray)

    def test_all_finite(self, simple_ols):
        from interlace import ols_dfbetas_qr

        dfb = ols_dfbetas_qr(simple_ols)
        assert np.all(np.isfinite(dfb))

    def test_all_finite_wide(self, wide_ols):
        from interlace import ols_dfbetas_qr

        dfb = ols_dfbetas_qr(wide_ols)
        assert np.all(np.isfinite(dfb))


class TestOlsDfbetasQrCorrectness:
    def test_matches_brute_force_loo(self, simple_ols):
        """QR-based result must match exact LOO-refit DFBETAS (R convention).

        Uses LOO sigma in denominator, matching R's influence.measures().
        Tolerance 1e-6 absolute since both paths use float64.
        """
        from interlace import ols_dfbetas_qr

        dfb_qr = ols_dfbetas_qr(simple_ols)
        dfb_loo = _brute_force_dfbetas(simple_ols)
        np.testing.assert_allclose(dfb_qr, dfb_loo, atol=1e-6)

    def test_intercept_column_included(self, simple_ols):
        """All p columns in design matrix get a DFBETAS column."""
        from interlace import ols_dfbetas_qr

        dfb = ols_dfbetas_qr(simple_ols)
        _, p = simple_ols.model.exog.shape
        assert dfb.shape[1] == p

    def test_influential_obs_has_large_dfbetas(self, simple_ols):
        """Manually add an outlier — it should produce the largest DFBETAS row."""
        from interlace import ols_dfbetas_qr

        X = simple_ols.model.exog.copy()
        y = simple_ols.model.endog.copy()

        # Add a high-leverage outlier at position 0
        X_out = X.copy()
        y_out = y.copy()
        X_out[0, 1] = 10.0  # extreme predictor value
        y_out[0] = -10.0  # extreme response
        model_out = sm.OLS(y_out, X_out).fit()

        dfb = ols_dfbetas_qr(model_out)
        max_abs_row = np.max(np.abs(dfb), axis=1)
        assert np.argmax(max_abs_row) == 0, "Outlier at index 0 should have max DFBETAS"


class TestOlsDfbetasQrInterface:
    def test_accepts_ols_results_wrapper(self, simple_ols):
        """Must accept statsmodels RegressionResultsWrapper directly."""
        from interlace import ols_dfbetas_qr

        # Should not raise
        dfb = ols_dfbetas_qr(simple_ols)
        assert dfb is not None

    def test_single_predictor(self):
        """Works with minimal model: intercept only."""
        from interlace import ols_dfbetas_qr

        rng = np.random.default_rng(0)
        y = rng.standard_normal(20)
        X = np.ones((20, 1))
        model = sm.OLS(y, X).fit()
        dfb = ols_dfbetas_qr(model)
        assert dfb.shape == (20, 1)
        assert np.all(np.isfinite(dfb))
