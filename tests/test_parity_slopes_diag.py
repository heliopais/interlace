"""Parity tests: interlace diagnostic quantities vs R lme4/HLMdiag, random slopes.

Uses the correlated slopes fixture (600 obs, 20 groups).
Model: y ~ x + (1 + x | g)

Acceptance criteria:
  - Conditional residuals: correlation > 0.999
  - Leverage H2 (ranef): correlation > 0.99
  - Cook's D: correlation > 0.95
  - augment.py returns a well-formed DataFrame for a slope model
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace.augment import hlm_augment
from interlace.influence import hlm_influence
from interlace.leverage import leverage
from interlace.residuals import hlm_resid

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "slopes_corr_data.csv")


@pytest.fixture(scope="module")
def r_results() -> dict:
    return json.loads((FIXTURES / "slopes_corr_r_results.json").read_text())


@pytest.fixture(scope="module")
def il_result(data):
    return interlace.fit("y ~ x", data=data, random=["(1 + x | g)"])


# --- residuals ---


def test_conditional_residuals_corr(il_result, r_results):
    resid_df = hlm_resid(il_result, type="conditional", full_data=False)
    il_resid = resid_df[".resid"].values
    r_resid = np.array(r_results["resid_cond"])
    corr = float(np.corrcoef(il_resid, r_resid)[0, 1])
    assert corr > 0.999, f"Conditional residual correlation={corr:.6f}"


def test_marginal_residuals_corr(il_result, r_results):
    resid_df = hlm_resid(il_result, type="marginal", full_data=False)
    il_resid = resid_df[".resid"].values
    r_resid = np.array(r_results["resid_marg"])
    corr = float(np.corrcoef(il_resid, r_resid)[0, 1])
    assert corr > 0.999, f"Marginal residual correlation={corr:.6f}"


# --- leverage ---


def test_leverage_ranef_corr(il_result, r_results):
    lev_df = leverage(il_result)
    il_h2 = lev_df["ranef"].values
    r_h2 = np.array(r_results["leverage"]["ranef"])
    # Filter out zeros from both (R may return 0 for some obs)
    mask = (np.abs(il_h2) > 1e-12) | (np.abs(r_h2) > 1e-12)
    corr = float(np.corrcoef(il_h2[mask], r_h2[mask])[0, 1])
    assert corr > 0.99, f"Leverage H2 correlation={corr:.4f}"


def test_leverage_fixef_nonneg(il_result):
    lev_df = leverage(il_result)
    assert (lev_df["fixef"] >= -1e-10).all()


def test_leverage_overall_equals_fixef_plus_ranef(il_result):
    lev_df = leverage(il_result)
    np.testing.assert_allclose(
        lev_df["overall"].values,
        lev_df["fixef"].values + lev_df["ranef"].values,
        atol=1e-10,
    )


# --- influence (Cook's D) ---


@pytest.fixture(scope="module")
def il_influence_df(il_result):
    return hlm_influence(il_result, level=1)


def test_cooksd_corr(il_influence_df, r_results):
    il_cd = il_influence_df["cooksd"].fillna(0).values
    r_cd = np.array(r_results["cooksd"])
    corr = float(np.corrcoef(il_cd, r_cd)[0, 1])
    assert corr > 0.95, f"Cook's D correlation={corr:.4f}"


def test_cooksd_nonneg(il_influence_df):
    assert (il_influence_df["cooksd"].fillna(0) >= -1e-10).all()


# --- augment ---


def test_augment_slopes_returns_dataframe(il_result, data):
    result = hlm_augment(il_result)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    for col in (".resid", ".fitted", "cooksd"):
        assert col in result.columns, f"missing column: {col}"
