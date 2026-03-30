"""Tests for interlace.ols(): formulaic-based OLS with HC3 robust SE.

Acceptance criteria (gpg-package-hs7):
  - params coefficients match numpy lstsq
  - hc3_bse() matches statsmodels get_robustcov_results('HC3').bse to 6 sig figs
  - predict(new_data) correctly evaluates formula on out-of-sample data
  - Works with Polars DataFrames natively
  - OLSResult exposes: params, resid, fittedvalues, normalized_cov_params,
    model.exog, model.endog, model.exog_names, model.formula, model.data.frame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
import statsmodels.formula.api as smf

from interlace.ols import OLSResult, ols

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data_pd():
    """Synthetic y ~ intercept + x1 + x2 dataset as pandas DataFrame."""
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 2, n)
    y = 5.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def synthetic_data_pl(synthetic_data_pd):
    """Same data as Polars DataFrame."""
    return pl.from_pandas(synthetic_data_pd)


@pytest.fixture(scope="module")
def ols_fit_pd(synthetic_data_pd):
    return ols("y ~ x1 + x2", synthetic_data_pd)


@pytest.fixture(scope="module")
def ols_fit_pl(synthetic_data_pl):
    return ols("y ~ x1 + x2", synthetic_data_pl)


@pytest.fixture(scope="module")
def statsmodels_fit(synthetic_data_pd):
    return smf.ols("y ~ x1 + x2", data=synthetic_data_pd).fit()


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_ols_result(self, ols_fit_pd):
        assert isinstance(ols_fit_pd, OLSResult)


# ---------------------------------------------------------------------------
# params attribute
# ---------------------------------------------------------------------------


class TestParams:
    def test_params_is_series(self, ols_fit_pd):
        assert isinstance(ols_fit_pd.params, pd.Series)

    def test_params_has_named_index(self, ols_fit_pd):
        assert "Intercept" in ols_fit_pd.params.index or "intercept" in [
            s.lower() for s in ols_fit_pd.params.index
        ]
        assert "x1" in ols_fit_pd.params.index
        assert "x2" in ols_fit_pd.params.index

    def test_params_match_numpy_lstsq(self, ols_fit_pd, synthetic_data_pd):
        df = synthetic_data_pd
        X = np.column_stack([np.ones(len(df)), df["x1"].values, df["x2"].values])
        y = df["y"].values
        beta_np, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(
            ols_fit_pd.params.values,
            beta_np,
            rtol=1e-10,
        )

    def test_params_polars_matches_pandas(self, ols_fit_pd, ols_fit_pl):
        np.testing.assert_allclose(
            ols_fit_pd.params.values,
            ols_fit_pl.params.values,
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# resid and fittedvalues
# ---------------------------------------------------------------------------


class TestResidualsAndFitted:
    def test_resid_shape(self, ols_fit_pd, synthetic_data_pd):
        assert ols_fit_pd.resid.shape == (len(synthetic_data_pd),)

    def test_fittedvalues_shape(self, ols_fit_pd, synthetic_data_pd):
        assert ols_fit_pd.fittedvalues.shape == (len(synthetic_data_pd),)

    def test_fitted_plus_resid_equals_y(self, ols_fit_pd, synthetic_data_pd):
        y = synthetic_data_pd["y"].values
        np.testing.assert_allclose(
            ols_fit_pd.fittedvalues + ols_fit_pd.resid, y, rtol=1e-10
        )


# ---------------------------------------------------------------------------
# normalized_cov_params
# ---------------------------------------------------------------------------


class TestNormalizedCovParams:
    def test_normalized_cov_params_shape(self, ols_fit_pd):
        p = len(ols_fit_pd.params)
        assert ols_fit_pd.normalized_cov_params.shape == (p, p)

    def test_normalized_cov_params_is_XtX_inv(self, ols_fit_pd, synthetic_data_pd):
        df = synthetic_data_pd
        X = np.column_stack([np.ones(len(df)), df["x1"].values, df["x2"].values])
        XtX_inv = np.linalg.inv(X.T @ X)
        np.testing.assert_allclose(ols_fit_pd.normalized_cov_params, XtX_inv, rtol=1e-8)

    def test_normalized_cov_params_matches_statsmodels(
        self, ols_fit_pd, statsmodels_fit
    ):
        np.testing.assert_allclose(
            ols_fit_pd.normalized_cov_params,
            statsmodels_fit.normalized_cov_params,
            rtol=1e-8,
        )


# ---------------------------------------------------------------------------
# hc3_bse
# ---------------------------------------------------------------------------


class TestHC3:
    def test_hc3_bse_shape(self, ols_fit_pd):
        p = len(ols_fit_pd.params)
        assert ols_fit_pd.hc3_bse().shape == (p,)

    def test_hc3_bse_positive(self, ols_fit_pd):
        assert np.all(ols_fit_pd.hc3_bse() > 0)

    def test_hc3_bse_matches_statsmodels_to_6_sig_figs(
        self, ols_fit_pd, statsmodels_fit
    ):
        sm_hc3 = statsmodels_fit.get_robustcov_results("HC3").bse
        our_hc3 = ols_fit_pd.hc3_bse()
        np.testing.assert_allclose(our_hc3, sm_hc3, rtol=1e-5)

    def test_hc3_bse_polars_matches_pandas(self, ols_fit_pd, ols_fit_pl):
        np.testing.assert_allclose(
            ols_fit_pd.hc3_bse(), ols_fit_pl.hc3_bse(), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# model attribute
# ---------------------------------------------------------------------------


class TestModelAttribute:
    def test_model_exog_shape(self, ols_fit_pd, synthetic_data_pd):
        n = len(synthetic_data_pd)
        assert ols_fit_pd.model.exog.shape == (n, 3)  # intercept + x1 + x2

    def test_model_endog_shape(self, ols_fit_pd, synthetic_data_pd):
        assert ols_fit_pd.model.endog.shape == (len(synthetic_data_pd),)

    def test_model_exog_names(self, ols_fit_pd):
        names = ols_fit_pd.model.exog_names
        assert isinstance(names, list)
        assert len(names) == 3
        assert "x1" in names
        assert "x2" in names

    def test_model_formula(self, ols_fit_pd):
        assert ols_fit_pd.model.formula == "y ~ x1 + x2"

    def test_model_data_frame_pandas(self, ols_fit_pd, synthetic_data_pd):
        assert ols_fit_pd.model.data.frame is synthetic_data_pd

    def test_model_data_frame_polars(self, ols_fit_pl, synthetic_data_pl):
        assert ols_fit_pl.model.data.frame is synthetic_data_pl


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_in_sample_matches_fittedvalues(
        self, ols_fit_pd, synthetic_data_pd
    ):
        preds = ols_fit_pd.predict(synthetic_data_pd)
        np.testing.assert_allclose(preds, ols_fit_pd.fittedvalues, rtol=1e-10)

    def test_predict_new_data_shape(self, ols_fit_pd):
        new_data = pd.DataFrame({"x1": [0.0, 1.0, -1.0], "x2": [0.0, 2.0, -2.0]})
        preds = ols_fit_pd.predict(new_data)
        assert preds.shape == (3,)

    def test_predict_new_data_values(self, ols_fit_pd):
        """Verify predict uses correct design matrix by checking against manual calc."""
        beta = ols_fit_pd.params.values
        new_data = pd.DataFrame({"x1": [1.0, -1.0], "x2": [0.5, -0.5]})
        preds = ols_fit_pd.predict(new_data)
        # Manual: [1, x1, x2] @ beta
        expected = np.array(
            [
                beta[0] + beta[1] * 1.0 + beta[2] * 0.5,
                beta[0] + beta[1] * (-1.0) + beta[2] * (-0.5),
            ]
        )
        np.testing.assert_allclose(preds, expected, rtol=1e-10)

    def test_predict_polars_new_data(self, ols_fit_pd):
        new_data_pd = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 2.0]})
        new_data_pl = pl.from_pandas(new_data_pd)
        preds_pd = ols_fit_pd.predict(new_data_pd)
        preds_pl = ols_fit_pd.predict(new_data_pl)
        np.testing.assert_allclose(preds_pd, preds_pl, rtol=1e-10)


# ---------------------------------------------------------------------------
# Polars native (no internal .to_pandas())
# ---------------------------------------------------------------------------


class TestPolarsNative:
    def test_ols_polars_no_pandas_conversion(self, synthetic_data_pl, monkeypatch):
        """Fitting on a Polars frame should not call .to_pandas() internally."""
        original_to_pandas = pl.DataFrame.to_pandas
        calls = []

        def spy_to_pandas(self, *args, **kwargs):
            calls.append(True)
            return original_to_pandas(self, *args, **kwargs)

        monkeypatch.setattr(pl.DataFrame, "to_pandas", spy_to_pandas)
        ols("y ~ x1 + x2", synthetic_data_pl)
        assert len(calls) == 0, "ols() called .to_pandas() on the Polars frame"
