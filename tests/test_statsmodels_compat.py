"""Tests for statsmodels compatibility attribute aliases.

Covers interlace-k39 and interlace-6x9:
- params, bse, tvalues, pvalues as aliases for fe_* counterparts
- llf_restricted attribute
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace


@pytest.fixture(scope="module")
def reml_result():
    rng = np.random.default_rng(42)
    n_groups, n_per = 10, 20
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.5, n_groups)
    y = (
        2.0
        + 0.8 * x
        + u[np.repeat(np.arange(n_groups), n_per)]
        + 0.5 * rng.standard_normal(n)
    )
    return interlace.fit(
        "y ~ x", data=pd.DataFrame({"y": y, "x": x, "g": g}), groups="g"
    )


@pytest.fixture(scope="module")
def ml_result():
    rng = np.random.default_rng(42)
    n_groups, n_per = 10, 20
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.5, n_groups)
    y = (
        2.0
        + 0.8 * x
        + u[np.repeat(np.arange(n_groups), n_per)]
        + 0.5 * rng.standard_normal(n)
    )
    return interlace.fit(
        "y ~ x", data=pd.DataFrame({"y": y, "x": x, "g": g}), groups="g", method="ML"
    )


# ---------------------------------------------------------------------------
# interlace-k39: params / bse / tvalues / pvalues aliases
# ---------------------------------------------------------------------------


class TestParamsAlias:
    def test_params_equals_fe_params(self, reml_result):
        pd.testing.assert_series_equal(reml_result.params, reml_result.fe_params)

    def test_params_has_correct_index(self, reml_result):
        assert "Intercept" in reml_result.params.index
        assert "x" in reml_result.params.index

    def test_params_is_series(self, reml_result):
        import pandas as pd

        assert isinstance(reml_result.params, pd.Series)


class TestBseAlias:
    def test_bse_equals_fe_bse(self, reml_result):
        pd.testing.assert_series_equal(reml_result.bse, reml_result.fe_bse)

    def test_bse_all_positive(self, reml_result):
        assert (reml_result.bse > 0).all()


class TestTvaluesAlias:
    def test_tvalues_equals_fe_tvalues(self, reml_result):
        pd.testing.assert_series_equal(reml_result.tvalues, reml_result.fe_tvalues)

    def test_tvalues_matches_params_over_bse(self, reml_result):
        expected = reml_result.params / reml_result.bse
        pd.testing.assert_series_equal(reml_result.tvalues, expected)


class TestPvaluesAlias:
    def test_pvalues_equals_fe_pvalues(self, reml_result):
        pd.testing.assert_series_equal(reml_result.pvalues, reml_result.fe_pvalues)

    def test_pvalues_in_range(self, reml_result):
        assert ((reml_result.pvalues >= 0) & (reml_result.pvalues <= 1)).all()


# ---------------------------------------------------------------------------
# interlace-6x9: llf_restricted
# ---------------------------------------------------------------------------


class TestLlfRestricted:
    def test_llf_restricted_is_llf_for_reml(self, reml_result):
        """For REML fits, llf_restricted == llf (the restricted log-likelihood)."""
        assert reml_result.method == "REML"
        assert reml_result.llf_restricted == reml_result.llf

    def test_llf_restricted_is_none_for_ml(self, ml_result):
        """For ML fits, there is no restricted log-likelihood."""
        assert ml_result.method == "ML"
        assert ml_result.llf_restricted is None

    def test_llf_restricted_is_float_for_reml(self, reml_result):
        assert isinstance(reml_result.llf_restricted, float)

    def test_llf_restricted_is_negative(self, reml_result):
        # Log-likelihoods for real data are always negative
        assert reml_result.llf_restricted < 0
