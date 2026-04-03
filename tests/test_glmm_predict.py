"""Tests for GLMMResult.predict() on link and response scales."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import interlace
from interlace.glmm_laplace import GLMMResult

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cbpp_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "glmm_cbpp_data.csv")


@pytest.fixture(scope="module")
def cbpp_fit(cbpp_data) -> GLMMResult:
    return interlace.glmer(
        formula="proportion ~ C(period)",
        data=cbpp_data,
        family="binomial",
        groups="herd",
        weights=cbpp_data["size"].values.astype(float),
    )


@pytest.fixture(scope="module")
def poisson_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "glmm_poisson_data.csv")


@pytest.fixture(scope="module")
def poisson_fit(poisson_data) -> GLMMResult:
    return interlace.glmer(
        formula="y ~ x",
        data=poisson_data,
        family="poisson",
        groups="group",
    )


# ---------------------------------------------------------------------------
# In-sample prediction
# ---------------------------------------------------------------------------


class TestInSamplePredict:
    """predict() with no newdata returns fitted values."""

    def test_binomial_insample_response(self, cbpp_fit):
        """In-sample response-scale predictions should be probabilities."""
        pred = cbpp_fit.predict()
        assert pred.shape == (cbpp_fit.nobs,)
        assert np.all(pred > 0)
        assert np.all(pred < 1)

    def test_binomial_insample_link(self, cbpp_fit):
        """In-sample link-scale predictions are log-odds (unbounded)."""
        pred = cbpp_fit.predict(type="link")
        assert pred.shape == (cbpp_fit.nobs,)
        # Link scale can be negative
        assert np.any(pred < 0)

    def test_poisson_insample_response(self, poisson_fit):
        """In-sample response-scale predictions should be positive counts."""
        pred = poisson_fit.predict()
        assert pred.shape == (poisson_fit.nobs,)
        assert np.all(pred > 0)

    def test_poisson_insample_link(self, poisson_fit):
        """In-sample link-scale predictions are log(mu)."""
        pred_link = poisson_fit.predict(type="link")
        pred_resp = poisson_fit.predict(type="response")
        assert_allclose(np.exp(pred_link), pred_resp, rtol=1e-10)

    def test_link_response_consistency(self, cbpp_fit):
        """linkinv(predict(type='link')) == predict(type='response')."""
        eta = cbpp_fit.predict(type="link")
        mu = cbpp_fit.predict(type="response")
        assert_allclose(cbpp_fit.family.linkinv(eta), mu, rtol=1e-10)


# ---------------------------------------------------------------------------
# New-data prediction
# ---------------------------------------------------------------------------


class TestNewDataPredict:
    """predict(newdata=...) with seen and unseen groups."""

    def test_newdata_seen_group(self, cbpp_fit, cbpp_data):
        """Prediction on known group levels uses BLUPs."""
        new = cbpp_data.head(5).copy()
        pred = cbpp_fit.predict(newdata=new)
        assert pred.shape == (5,)
        assert np.all(np.isfinite(pred))
        # Response scale: probabilities
        assert np.all(pred > 0)
        assert np.all(pred < 1)

    def test_newdata_unseen_group(self, cbpp_fit, cbpp_data):
        """Unseen group levels should shrink RE to 0 (population mean)."""
        new = cbpp_data.head(3).copy()
        new["herd"] = 999  # unseen level
        pred_new = cbpp_fit.predict(newdata=new, type="link")
        # Should equal fixed-effects-only prediction
        pred_fe = cbpp_fit.predict(newdata=new, include_re=False, type="link")
        assert_allclose(pred_new, pred_fe, atol=1e-10)

    def test_newdata_link_scale(self, cbpp_fit, cbpp_data):
        """New-data prediction on link scale."""
        new = cbpp_data.head(3).copy()
        pred = cbpp_fit.predict(newdata=new, type="link")
        assert pred.shape == (3,)
        assert np.all(np.isfinite(pred))

    def test_include_re_false(self, cbpp_fit, cbpp_data):
        """include_re=False gives population-level predictions only."""
        pred_with = cbpp_fit.predict(newdata=cbpp_data.head(5), type="link")
        pred_without = cbpp_fit.predict(
            newdata=cbpp_data.head(5), include_re=False, type="link"
        )
        # With RE should differ from without (herds have non-zero BLUPs)
        assert not np.allclose(pred_with, pred_without)

    def test_poisson_newdata(self, poisson_fit, poisson_data):
        """Poisson prediction on new data."""
        new = poisson_data.head(10).copy()
        pred = poisson_fit.predict(newdata=new)
        assert pred.shape == (10,)
        assert np.all(pred > 0)  # response scale: positive


# ---------------------------------------------------------------------------
# fittedvalues attribute
# ---------------------------------------------------------------------------


class TestFittedValues:
    """GLMMResult should have fittedvalues on the response scale."""

    def test_has_fittedvalues(self, cbpp_fit):
        assert hasattr(cbpp_fit, "fittedvalues")
        assert cbpp_fit.fittedvalues.shape == (cbpp_fit.nobs,)

    def test_fittedvalues_are_response_scale(self, cbpp_fit):
        """fittedvalues should be on response scale (probabilities)."""
        assert np.all(cbpp_fit.fittedvalues > 0)
        assert np.all(cbpp_fit.fittedvalues < 1)

    def test_predict_matches_fittedvalues(self, cbpp_fit):
        """In-sample predict() should equal fittedvalues."""
        assert_allclose(cbpp_fit.predict(), cbpp_fit.fittedvalues, atol=1e-10)
