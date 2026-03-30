"""Tests for interlace.quantreg(): LP-based quantile regression.

Acceptance criteria (gpg-package-ig1):
  - params coefficients match statsmodels QuantReg to 8 sig figs
  - ker_se() delegates to quantreg_ker_se() and returns same values
  - Works with Polars DataFrames natively
  - predict() handles new data with formula re-evaluation
  - QuantRegResult exposes: params, resid, fittedvalues, tau, predict(), ker_se()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
import statsmodels.formula.api as smf

from interlace.quantreg import QuantRegResult, quantreg, quantreg_ker_se


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data_pd():
    """Synthetic y ~ intercept + male dataset as pandas DataFrame."""
    rng = np.random.default_rng(42)
    n = 400
    male = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    y = 10_000 + 3_000 * male + rng.normal(0, 2_000, n)
    return pd.DataFrame({"y": y, "male": male})


@pytest.fixture(scope="module")
def synthetic_data_pl(synthetic_data_pd):
    return pl.from_pandas(synthetic_data_pd)


@pytest.fixture(scope="module")
def qr_fit_pd(synthetic_data_pd):
    return quantreg("y ~ male", synthetic_data_pd, tau=0.5)


@pytest.fixture(scope="module")
def qr_fit_pl(synthetic_data_pl):
    return quantreg("y ~ male", synthetic_data_pl, tau=0.5)


@pytest.fixture(scope="module")
def statsmodels_fit(synthetic_data_pd):
    return smf.quantreg("y ~ male", data=synthetic_data_pd).fit(q=0.5, disp=False)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_quantreg_result(self, qr_fit_pd):
        assert isinstance(qr_fit_pd, QuantRegResult)


# ---------------------------------------------------------------------------
# params attribute
# ---------------------------------------------------------------------------


class TestParams:
    def test_params_is_series(self, qr_fit_pd):
        assert isinstance(qr_fit_pd.params, pd.Series)

    def test_params_has_named_index(self, qr_fit_pd):
        assert "male" in qr_fit_pd.params.index

    def test_params_get_method(self, qr_fit_pd):
        val = qr_fit_pd.params.get("male")
        assert val is not None
        assert isinstance(float(val), float)

    def test_params_optimal_objective_matches_statsmodels(
        self, qr_fit_pd, statsmodels_fit, synthetic_data_pd
    ):
        """The LP objective value must equal statsmodels (both are optimal).

        QR LP can have multiple optimal solutions when residuals tie at zero,
        so we compare objective values rather than exact coefficients.
        """
        tau = qr_fit_pd.tau
        y = synthetic_data_pd["y"].values

        def qr_obj(params_vals):
            resid = y - params_vals[0] - params_vals[1] * synthetic_data_pd["male"].values
            return tau * np.sum(np.maximum(resid, 0)) + (1 - tau) * np.sum(
                np.maximum(-resid, 0)
            )

        obj_ours = qr_obj(qr_fit_pd.params.values)
        obj_sm = qr_obj(statsmodels_fit.params.values)
        np.testing.assert_allclose(obj_ours, obj_sm, rtol=1e-8)

    def test_params_polars_matches_pandas(self, qr_fit_pd, qr_fit_pl):
        np.testing.assert_allclose(
            qr_fit_pd.params.values,
            qr_fit_pl.params.values,
            rtol=1e-7,
        )


# ---------------------------------------------------------------------------
# resid and fittedvalues
# ---------------------------------------------------------------------------


class TestResidualsAndFitted:
    def test_resid_shape(self, qr_fit_pd, synthetic_data_pd):
        assert qr_fit_pd.resid.shape == (len(synthetic_data_pd),)

    def test_fittedvalues_shape(self, qr_fit_pd, synthetic_data_pd):
        assert qr_fit_pd.fittedvalues.shape == (len(synthetic_data_pd),)

    def test_fitted_plus_resid_equals_y(self, qr_fit_pd, synthetic_data_pd):
        y = synthetic_data_pd["y"].values
        np.testing.assert_allclose(
            qr_fit_pd.fittedvalues + qr_fit_pd.resid, y, rtol=1e-10
        )


# ---------------------------------------------------------------------------
# tau attribute
# ---------------------------------------------------------------------------


class TestTau:
    def test_tau_stored(self, qr_fit_pd):
        assert qr_fit_pd.tau == 0.5

    def test_tau_other_quantile(self, synthetic_data_pd):
        fit = quantreg("y ~ male", synthetic_data_pd, tau=0.75)
        assert fit.tau == 0.75

    def test_different_tau_different_params(self, synthetic_data_pd):
        fit_50 = quantreg("y ~ male", synthetic_data_pd, tau=0.5)
        fit_75 = quantreg("y ~ male", synthetic_data_pd, tau=0.75)
        assert not np.allclose(fit_50.params.values, fit_75.params.values)


# ---------------------------------------------------------------------------
# ker_se
# ---------------------------------------------------------------------------


class TestKerSe:
    def test_ker_se_shape(self, qr_fit_pd):
        p = len(qr_fit_pd.params)
        assert qr_fit_pd.ker_se().shape == (p,)

    def test_ker_se_positive(self, qr_fit_pd):
        assert np.all(qr_fit_pd.ker_se() > 0)

    def test_ker_se_delegates_to_quantreg_ker_se(self, qr_fit_pd):
        """ker_se() must return identical values to quantreg_ker_se() directly."""
        expected = quantreg_ker_se(
            qr_fit_pd.resid,
            qr_fit_pd._X,
            tau=qr_fit_pd.tau,
            hs=True,
        )
        np.testing.assert_array_equal(qr_fit_pd.ker_se(hs=True), expected)

    def test_ker_se_hs_false(self, qr_fit_pd):
        expected = quantreg_ker_se(
            qr_fit_pd.resid,
            qr_fit_pd._X,
            tau=qr_fit_pd.tau,
            hs=False,
        )
        np.testing.assert_array_equal(qr_fit_pd.ker_se(hs=False), expected)

    def test_ker_se_polars_matches_pandas(self, qr_fit_pd, qr_fit_pl):
        np.testing.assert_allclose(
            qr_fit_pd.ker_se(), qr_fit_pl.ker_se(), rtol=1e-7
        )


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_in_sample_matches_fittedvalues(self, qr_fit_pd, synthetic_data_pd):
        preds = qr_fit_pd.predict(synthetic_data_pd)
        np.testing.assert_allclose(preds, qr_fit_pd.fittedvalues, rtol=1e-10)

    def test_predict_new_data_shape(self, qr_fit_pd):
        new_data = pd.DataFrame({"male": [0.0, 1.0]})
        preds = qr_fit_pd.predict(new_data)
        assert preds.shape == (2,)

    def test_predict_new_data_values(self, qr_fit_pd):
        beta = qr_fit_pd.params.values
        new_data = pd.DataFrame({"male": [1.0, 0.0]})
        preds = qr_fit_pd.predict(new_data)
        expected = np.array([
            beta[0] + beta[1] * 1.0,
            beta[0] + beta[1] * 0.0,
        ])
        np.testing.assert_allclose(preds, expected, rtol=1e-10)

    def test_predict_polars_new_data(self, qr_fit_pd):
        new_data_pd = pd.DataFrame({"male": [0.0, 1.0]})
        new_data_pl = pl.from_pandas(new_data_pd)
        preds_pd = qr_fit_pd.predict(new_data_pd)
        preds_pl = qr_fit_pd.predict(new_data_pl)
        np.testing.assert_allclose(preds_pd, preds_pl, rtol=1e-10)


# ---------------------------------------------------------------------------
# Polars native (no internal .to_pandas())
# ---------------------------------------------------------------------------


class TestPolarsNative:
    def test_quantreg_polars_no_pandas_conversion(
        self, synthetic_data_pl, monkeypatch
    ):
        """Fitting on a Polars frame should not call .to_pandas() internally."""
        original_to_pandas = pl.DataFrame.to_pandas
        calls = []

        def spy_to_pandas(self, *args, **kwargs):
            calls.append(True)
            return original_to_pandas(self, *args, **kwargs)

        monkeypatch.setattr(pl.DataFrame, "to_pandas", spy_to_pandas)
        quantreg("y ~ male", synthetic_data_pl, tau=0.5)
        assert len(calls) == 0, "quantreg() called .to_pandas() on the Polars frame"
