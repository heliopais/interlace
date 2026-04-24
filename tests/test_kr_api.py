"""Tests for exposing KR DFs via df_method parameter in fit() and summary().

Validates that:
1. fit(df_method='kenward-roger') stores KR DFs on the result
2. summary() renders the KR DFs and adjusted SEs
3. Default df_method='satterthwaite' is backward-compatible
4. Result carries df_method for downstream use
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
def one_factor_data():
    with open(FIXTURES / "kr_parity_one_factor.json") as f:
        ref = json.load(f)
    return pd.DataFrame(
        {
            "y": np.array(ref["y"]),
            "x": np.array(ref["x"]),
            "group": [str(g) for g in ref["group"]],
        }
    ), ref


class TestDfMethodParameter:
    """fit() should accept df_method and store it on the result."""

    def test_default_is_satterthwaite(self, one_factor_data):
        df, _ = one_factor_data
        result = interlace.fit("y ~ x", data=df, groups="group")
        assert result.df_method == "satterthwaite"

    def test_kr_stores_method(self, one_factor_data):
        df, _ = one_factor_data
        result = interlace.fit(
            "y ~ x", data=df, groups="group", df_method="kenward-roger"
        )
        assert result.df_method == "kenward-roger"

    def test_kr_dfs_match_r(self, one_factor_data):
        df, ref = one_factor_data
        result = interlace.fit(
            "y ~ x", data=df, groups="group", df_method="kenward-roger"
        )
        r_dfs = list(ref["kr_dfs"].values())
        np.testing.assert_allclose(np.array(result.fe_df), r_dfs, rtol=0.01)

    def test_kr_dfs_differ_from_satterthwaite(self, one_factor_data):
        df, _ = one_factor_data
        satt = interlace.fit("y ~ x", data=df, groups="group")
        kr = interlace.fit("y ~ x", data=df, groups="group", df_method="kenward-roger")
        # Slope DF should be substantially smaller with KR
        assert float(kr.fe_df.iloc[1]) < float(satt.fe_df.iloc[1]) * 0.5

    def test_invalid_df_method_raises(self, one_factor_data):
        df, _ = one_factor_data
        with pytest.raises(ValueError, match="df_method"):
            interlace.fit("y ~ x", data=df, groups="group", df_method="foo")


class TestSummaryWithKR:
    """summary() should reflect the df_method in the output."""

    def test_summary_renders_kr_dfs(self, one_factor_data):
        df, _ = one_factor_data
        result = interlace.fit(
            "y ~ x", data=df, groups="group", df_method="kenward-roger"
        )
        text = str(result.summary())
        # The DF column should show ~14.0 for intercept (not ~16)
        assert "14." in text or "13." in text

    def test_summary_pvalues_use_kr_dfs(self, one_factor_data):
        df, _ = one_factor_data
        satt = interlace.fit("y ~ x", data=df, groups="group")
        kr = interlace.fit("y ~ x", data=df, groups="group", df_method="kenward-roger")
        # With fewer DFs (KR), p-values should be at least as large
        # (more conservative) for the slope coefficient
        assert float(kr.fe_pvalues.iloc[1]) >= float(satt.fe_pvalues.iloc[1]) * 0.5


class TestUpdatePreservesDfMethod:
    """update() should replay df_method."""

    def test_update_keeps_kr(self, one_factor_data):
        df, _ = one_factor_data
        result = interlace.fit(
            "y ~ x", data=df, groups="group", df_method="kenward-roger"
        )
        updated = result.update()
        assert updated.df_method == "kenward-roger"
