"""Tests for interlace.residuals.hlm_resid.

Acceptance criteria:
  - Returns DataFrame with .resid and .fitted columns
  - Marginal residuals match statsmodels to correlation > 0.999
  - Conditional residuals match statsmodels to correlation > 0.999
  - Works with both CrossedLMEResult and statsmodels MixedLMResults
  - full_data=True appends original data columns
  - standardized=True scales by sqrt(scale)
  - Group-level (level != 1) returns random effects DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.residuals import hlm_resid


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_groups, n_per = 20, 10
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.2, n_groups)
    eps = rng.normal(0, 0.8, n)
    y = 2.0 + 1.5 * x + u[group_ids] + eps
    return pd.DataFrame({"y": y, "x": x, "group": group_ids.astype(str)})


@pytest.fixture(scope="module")
def models(data):
    sm = MixedLM.from_formula("y ~ x", groups="group", data=data).fit(reml=True)
    il = interlace.fit("y ~ x", data=data, groups="group")
    return sm, il


# --- structure ---


@pytest.mark.parametrize("resid_type", ["marginal", "conditional"])
def test_returns_dataframe_with_required_cols(models, resid_type):
    sm, il = models
    for model in (sm, il):
        result = hlm_resid(model, type=resid_type, full_data=False)
        assert isinstance(result, pd.DataFrame)
        assert ".resid" in result.columns
        assert ".fitted" in result.columns


def test_full_data_appends_original_columns(models, data):
    sm, il = models
    for model in (sm, il):
        result = hlm_resid(model, type="marginal", full_data=True)
        assert len(result) == len(data)
        assert "y" in result.columns
        assert "x" in result.columns
        assert ".resid" in result.columns


def test_invalid_type_raises(models):
    _, il = models
    with pytest.raises(ValueError, match="type must be"):
        hlm_resid(il, type="invalid")


# --- marginal residuals ---


def test_marginal_resid_match_across_models(models):
    sm, il = models
    sm_r = hlm_resid(sm, type="marginal", full_data=False)[".resid"].values
    il_r = hlm_resid(il, type="marginal", full_data=False)[".resid"].values
    corr = np.corrcoef(sm_r, il_r)[0, 1]
    assert corr > 0.999, f"Marginal residual correlation={corr:.6f}"


def test_marginal_fitted_is_xbeta(models, data):
    """Marginal fitted values should equal X*beta (no RE contribution)."""
    _, il = models
    result = hlm_resid(il, type="marginal", full_data=False)
    xbeta = il.model.exog @ il.fe_params.values
    np.testing.assert_allclose(result[".fitted"].values, xbeta, rtol=1e-10)


# --- conditional residuals ---


def test_conditional_resid_match_across_models(models):
    sm, il = models
    sm_r = hlm_resid(sm, type="conditional", full_data=False)[".resid"].values
    il_r = hlm_resid(il, type="conditional", full_data=False)[".resid"].values
    corr = np.corrcoef(sm_r, il_r)[0, 1]
    assert corr > 0.999, f"Conditional residual correlation={corr:.6f}"


def test_conditional_smaller_variance_than_marginal(models):
    """Conditioning on RE should reduce residual variance."""
    _, il = models
    marg = hlm_resid(il, type="marginal", full_data=False)[".resid"].var()
    cond = hlm_resid(il, type="conditional", full_data=False)[".resid"].var()
    assert cond < marg, "Conditional variance should be less than marginal"


# --- standardized ---


def test_standardized_scales_by_sqrt_scale(models):
    _, il = models
    raw = hlm_resid(il, type="marginal", standardized=False, full_data=False)[
        ".resid"
    ].values
    std = hlm_resid(il, type="marginal", standardized=True, full_data=False)[
        ".resid"
    ].values
    expected = raw / np.sqrt(il.scale)
    np.testing.assert_allclose(std, expected, rtol=1e-10)


# --- group-level residuals ---


def test_group_level_returns_ranef_dataframe(models):
    sm, il = models
    sm_re = hlm_resid(sm, level="group")
    il_re = hlm_resid(il, level="group")
    assert isinstance(sm_re, pd.DataFrame)
    assert isinstance(il_re, pd.DataFrame)
    assert any(c.startswith(".ranef") for c in sm_re.columns)
    assert any(c.startswith(".ranef") for c in il_re.columns)


def test_group_level_ranef_values_match(models):
    sm, il = models
    sm_re = hlm_resid(sm, level="group").set_index("group").filter(like=".ranef")
    il_re = hlm_resid(il, level="group").set_index("group").filter(like=".ranef")
    common = sm_re.index.intersection(il_re.index)
    sm_vals = sm_re.loc[common].values.ravel()
    il_vals = il_re.loc[common].values.ravel()
    corr = np.corrcoef(sm_vals, il_vals)[0, 1]
    assert corr > 0.99, f"Group RE correlation={corr:.4f}"


# --- random slopes residuals ---


@pytest.fixture(scope="module")
def slope_data_resid() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    n_groups, n_per = 8, 6
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    b0 = rng.normal(0, 0.8, n_groups)
    b1 = rng.normal(0, 0.4, n_groups)
    eps = rng.normal(0, 0.5, n)
    y = 1.0 + 0.5 * x + b0[g.astype(int)] + b1[g.astype(int)] * x + eps
    return pd.DataFrame({"y": y, "x": x, "g": g})


@pytest.fixture(scope="module")
def slope_model_resid(slope_data_resid: pd.DataFrame):
    return interlace.fit("y ~ x", data=slope_data_resid, random=["(1 + x | g)"])


def test_marginal_resid_slopes_length(slope_model_resid, slope_data_resid):
    result = hlm_resid(slope_model_resid, type="marginal", full_data=False)
    assert len(result) == len(slope_data_resid)
    assert ".resid" in result.columns


def test_conditional_resid_slopes_length(slope_model_resid, slope_data_resid):
    result = hlm_resid(slope_model_resid, type="conditional", full_data=False)
    assert len(result) == len(slope_data_resid)
    assert ".resid" in result.columns


def test_group_level_resid_slopes_returns_dataframe(
    slope_model_resid, slope_data_resid
):
    result = hlm_resid(slope_model_resid, level="g")
    assert isinstance(result, pd.DataFrame)
    assert "g" in result.columns
    assert any(c.startswith(".ranef") for c in result.columns)
    assert len(result) == slope_data_resid["g"].nunique()


def test_group_level_resid_slopes_has_per_term_columns(slope_model_resid):
    """Group-level output must have one .ranef column per RE term."""
    result = hlm_resid(slope_model_resid, level="g")
    ranef_cols = [c for c in result.columns if c.startswith(".ranef")]
    # (1+x|g) has 2 terms → 2 ranef columns
    assert len(ranef_cols) == 2, f"expected 2 ranef columns, got: {ranef_cols}"
