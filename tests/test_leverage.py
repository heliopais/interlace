"""Tests for interlace.leverage.leverage.

Acceptance criteria:
  - Returns DataFrame with overall, fixef, ranef, ranef.uc columns
  - Works with both CrossedLMEResult and statsmodels MixedLMResults
  - fixef values from both correlate > 0.99
  - overall values from both correlate > 0.99
  - overall ≈ fixef + ranef (within floating-point tolerance)
  - All leverage values are non-negative
  - For crossed RE: trace(h_fixef) == p (OLS hat, not block-diagonal GLS)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.leverage import leverage


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


def test_returns_dataframe_with_required_cols(models):
    sm, il = models
    for model in (sm, il):
        result = leverage(model)
        assert isinstance(result, pd.DataFrame)
        for col in ("overall", "fixef", "ranef", "ranef.uc"):
            assert col in result.columns, f"missing column: {col}"


def test_length_equals_nobs(models, data):
    sm, il = models
    for model in (sm, il):
        result = leverage(model)
        assert len(result) == len(data)


def test_values_nonnegative(models):
    sm, il = models
    for model in (sm, il):
        result = leverage(model)
        for col in ("overall", "fixef", "ranef", "ranef.uc"):
            assert (result[col] >= -1e-10).all(), f"negative values in {col}"


def test_overall_equals_fixef_plus_ranef(models):
    sm, il = models
    for model in (sm, il):
        result = leverage(model)
        np.testing.assert_allclose(
            result["overall"].values,
            result["fixef"].values + result["ranef"].values,
            atol=1e-10,
        )


# --- parity across model types ---


def test_fixef_leverage_match_across_models(models):
    sm, il = models
    sm_h1 = leverage(sm)["fixef"].values
    il_h1 = leverage(il)["fixef"].values
    corr = np.corrcoef(sm_h1, il_h1)[0, 1]
    assert corr > 0.99, f"fixef leverage correlation={corr:.4f}"


def test_overall_leverage_match_across_models(models):
    sm, il = models
    sm_h = leverage(sm)["overall"].values
    il_h = leverage(il)["overall"].values
    corr = np.corrcoef(sm_h, il_h)[0, 1]
    assert corr > 0.99, f"overall leverage correlation={corr:.4f}"


# --- crossed RE: fixef trace must equal p (OLS hat) ---


@pytest.fixture(scope="module")
def crossed_model() -> Any:
    """Fit a truly crossed RE model: firm X dept (not nested)."""
    rng = np.random.default_rng(99)
    n_firms, n_depts = 15, 8
    n = 200
    firm_idx = rng.integers(0, n_firms, size=n)
    dept_idx = rng.integers(0, n_depts, size=n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    u_firm = rng.normal(0, 1.0, n_firms)
    u_dept = rng.normal(0, 0.8, n_depts)
    eps = rng.normal(0, 0.5, n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + u_firm[firm_idx] + u_dept[dept_idx] + eps
    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm": firm_idx.astype(str),
            "dept": dept_idx.astype(str),
        }
    )
    return interlace.fit("y ~ x1 + x2", data=df, groups=["firm", "dept"])


def test_crossed_fixef_trace_equals_p(crossed_model: Any) -> None:
    """trace(h_fixef) must equal p (number of fixed-effect columns) for crossed RE."""
    result = leverage(crossed_model)
    X = crossed_model.model.exog
    p = X.shape[1]
    trace_h1 = result["fixef"].sum()
    np.testing.assert_allclose(
        trace_h1, p, atol=1e-8, err_msg=f"trace(H1)={trace_h1:.4f}, expected p={p}"
    )


def test_crossed_fixef_equals_ols_hat(crossed_model: Any) -> None:
    """fixef leverage must equal the OLS hat diagonal for crossed RE."""
    result = leverage(crossed_model)
    X = crossed_model.model.exog
    XtX_inv = np.linalg.pinv(X.T @ X)
    h_ols = np.sum((X @ XtX_inv) * X, axis=1)
    np.testing.assert_allclose(result["fixef"].values, h_ols, atol=1e-10)


def test_crossed_overall_equals_fixef_plus_ranef(crossed_model: Any) -> None:
    result = leverage(crossed_model)
    np.testing.assert_allclose(
        result["overall"].values,
        result["fixef"].values + result["ranef"].values,
        atol=1e-10,
    )


# --- random slopes leverage ---


@pytest.fixture(scope="module")
def slope_data_lev() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n_groups, n_per = 10, 8
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    b0 = rng.normal(0, 0.8, n_groups)
    b1 = rng.normal(0, 0.4, n_groups)
    eps = rng.normal(0, 0.5, n)
    y = 1.0 + 0.5 * x + b0[g.astype(int)] + b1[g.astype(int)] * x + eps
    return pd.DataFrame({"y": y, "x": x, "g": g})


@pytest.fixture(scope="module")
def slope_model_lev(slope_data_lev: pd.DataFrame) -> Any:
    return interlace.fit("y ~ x", data=slope_data_lev, random=["(1 + x | g)"])


def test_leverage_slopes_returns_required_cols(slope_model_lev: Any) -> None:
    result = leverage(slope_model_lev)
    assert isinstance(result, pd.DataFrame)
    for col in ("overall", "fixef", "ranef", "ranef.uc"):
        assert col in result.columns, f"missing column: {col}"


def test_leverage_slopes_values_nonneg(slope_model_lev: Any) -> None:
    result = leverage(slope_model_lev)
    for col in ("overall", "fixef", "ranef", "ranef.uc"):
        assert (result[col] >= -1e-10).all(), f"negative values in {col}"


def test_leverage_slopes_overall_equals_fixef_plus_ranef(slope_model_lev: Any) -> None:
    result = leverage(slope_model_lev)
    np.testing.assert_allclose(
        result["overall"].values,
        result["fixef"].values + result["ranef"].values,
        atol=1e-10,
    )


def test_leverage_slopes_ranef_nonzero(slope_model_lev: Any) -> None:
    """Random-effects leverage should be non-trivially nonzero for slope models."""
    result = leverage(slope_model_lev)
    assert result["ranef"].sum() > 0.01
