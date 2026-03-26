"""Parity test: interlace vs R lme4, crossed factors with slope on g1.

Model:  y ~ x + (1 + x | g1) + (1 | g2)
Data:   600 obs, 15 g1 levels × 8 g2 levels
Oracle: tests/fixtures/slopes_crossed_r_results.json (gen_slopes_crossed.R)

Acceptance criteria (from CLAUDE.md):
  - Fixed effects abs_diff < 1e-4
  - Variance components rel_diff < 5%
  - BLUP correlation > 0.99
  - Conditional residual correlation > 0.999
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "slopes_crossed_data.csv")


@pytest.fixture(scope="module")
def r_results() -> dict:
    return json.loads((FIXTURES / "slopes_crossed_r_results.json").read_text())


@pytest.fixture(scope="module")
def il_result(data):
    return interlace.fit("y ~ x", data=data, random=["(1 + x | g1)", "(1 | g2)"])


def test_fixed_effects_match(il_result, r_results):
    r_fe = r_results["fe_params"]
    name_map = {"(Intercept)": "Intercept", "x": "x"}
    for r_name, il_name in name_map.items():
        diff = abs(il_result.fe_params[il_name] - r_fe[r_name])
        assert diff < 1e-4, (
            f"Fixed effect '{il_name}' abs_diff={diff:.2e} "
            f"(interlace={il_result.fe_params[il_name]:.6f}, R={r_fe[r_name]:.6f})"
        )


def test_variance_components_g1_diagonal(il_result, r_results):
    vc = il_result.variance_components["g1"]
    r_cov = r_results["cov_g1"]
    for term in ["(Intercept)", "x"]:
        il_val = float(vc.loc[term, term])
        r_val = float(r_cov[term][term])
        rel_diff = abs(il_val - r_val) / r_val
        assert rel_diff < 0.05, (
            f"g1 VC diagonal '{term}' rel_diff={rel_diff:.2%} "
            f"(interlace={il_val:.6f}, R={r_val:.6f})"
        )


def test_variance_components_g2(il_result, r_results):
    il_var = il_result.variance_components["g2"]
    r_var = r_results["var_g2"]
    rel_diff = abs(il_var - r_var) / r_var
    assert rel_diff < 0.05, (
        f"g2 variance rel_diff={rel_diff:.2%} (interlace={il_var:.6f}, R={r_var:.6f})"
    )


def test_residual_variance(il_result, r_results):
    r_var = r_results["var_resid"]
    rel_diff = abs(il_result.scale - r_var) / r_var
    assert rel_diff < 0.05, (
        f"Residual variance rel_diff={rel_diff:.2%} "
        f"(interlace={il_result.scale:.6f}, R={r_var:.6f})"
    )


def test_blups_g1_intercept_correlated(il_result, r_results):
    r_blups = r_results["blups_g1"]
    labels = sorted(r_blups.keys())
    r_arr = np.array([r_blups[g]["(Intercept)"] for g in labels])
    il_arr = il_result.random_effects["g1"].loc[labels, "(Intercept)"].to_numpy()
    corr = np.corrcoef(r_arr, il_arr)[0, 1]
    assert corr > 0.99, f"g1 intercept BLUP correlation={corr:.4f} < 0.99"


def test_blups_g1_slope_correlated(il_result, r_results):
    r_blups = r_results["blups_g1"]
    labels = sorted(r_blups.keys())
    r_arr = np.array([r_blups[g]["x"] for g in labels])
    il_arr = il_result.random_effects["g1"].loc[labels, "x"].to_numpy()
    corr = np.corrcoef(r_arr, il_arr)[0, 1]
    assert corr > 0.99, f"g1 slope BLUP correlation={corr:.4f} < 0.99"


def test_blups_g2_correlated(il_result, r_results):
    r_blups = r_results["blups_g2"]
    labels = sorted(r_blups.keys())
    r_arr = np.array([r_blups[g] for g in labels])
    il_arr = il_result.random_effects["g2"].loc[labels].to_numpy()
    corr = np.corrcoef(r_arr, il_arr)[0, 1]
    assert corr > 0.99, f"g2 BLUP correlation={corr:.4f} < 0.99"


def test_residuals_correlated(il_result, r_results):
    r_resid = np.array(r_results["resid_cond"])
    corr = np.corrcoef(il_result.resid, r_resid)[0, 1]
    assert corr > 0.999, f"Residual correlation={corr:.6f} < 0.999"
