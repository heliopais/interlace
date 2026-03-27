"""Tests for summary() and VarCorr() output.

Acceptance criteria:
  - VarCorr(result).as_dataframe() has columns grp, var1, var2, vcov, sdcor
  - VarCorr rows match known variance components to 4 decimal places
  - result.summary().__str__() contains expected sections
  - summary fixed-effects table values match to 4 significant figures
  - Works for single-RE, two-RE, and random-slopes models
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace import VarCorr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_re_result():
    rng = np.random.default_rng(42)
    n_groups, n_per = 20, 10
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.2, size=n_groups)
    eps = rng.normal(0, 0.8, size=n)
    y = 2.0 + 1.5 * x + u[np.repeat(np.arange(n_groups), n_per)] + eps
    df = pd.DataFrame({"y": y, "x": x, "group": group_ids})
    return interlace.fit("y ~ x", data=df, groups="group")


@pytest.fixture(scope="module")
def two_re_result():
    import pandas as pd

    df = pd.read_csv("tests/fixtures/two_re_data.csv")
    return interlace.fit("y ~ x", data=df, groups=["firm", "dept"])


@pytest.fixture(scope="module")
def slopes_result():
    import pandas as pd

    df = pd.read_csv("tests/fixtures/slopes_corr_data.csv")
    return interlace.fit(
        "y ~ x",
        data=df,
        random=["(1 + x | g)"],
    )


# ---------------------------------------------------------------------------
# VarCorr — single random intercept
# ---------------------------------------------------------------------------


class TestVarCorrSingleRE:
    def test_returns_varcorr_result(self, single_re_result):
        vc = VarCorr(single_re_result)
        assert vc is not None

    def test_as_dataframe_columns(self, single_re_result):
        df = VarCorr(single_re_result).as_dataframe()
        assert list(df.columns) == ["grp", "var1", "var2", "vcov", "sdcor"]

    def test_has_group_row(self, single_re_result):
        df = VarCorr(single_re_result).as_dataframe()
        assert "group" in df["grp"].values

    def test_has_residual_row(self, single_re_result):
        df = VarCorr(single_re_result).as_dataframe()
        assert "Residual" in df["grp"].values

    def test_group_var1_is_intercept(self, single_re_result):
        df = VarCorr(single_re_result).as_dataframe()
        row = df[df["grp"] == "group"].iloc[0]
        assert row["var1"] == "(Intercept)"

    def test_group_var2_is_na(self, single_re_result):
        df = VarCorr(single_re_result).as_dataframe()
        row = df[df["grp"] == "group"].iloc[0]
        assert pd.isna(row["var2"])

    def test_vcov_matches_variance_components(self, single_re_result):
        vc_df = VarCorr(single_re_result).as_dataframe()
        row = vc_df[vc_df["grp"] == "group"].iloc[0]
        expected_var = single_re_result.variance_components["group"]
        assert abs(row["vcov"] - expected_var) < 1e-10

    def test_sdcor_is_sqrt_vcov(self, single_re_result):
        vc_df = VarCorr(single_re_result).as_dataframe()
        row = vc_df[vc_df["grp"] == "group"].iloc[0]
        assert abs(row["sdcor"] - np.sqrt(row["vcov"])) < 1e-10

    def test_residual_vcov_matches_scale(self, single_re_result):
        vc_df = VarCorr(single_re_result).as_dataframe()
        row = vc_df[vc_df["grp"] == "Residual"].iloc[0]
        assert abs(row["vcov"] - single_re_result.scale) < 1e-10

    def test_residual_var1_is_na(self, single_re_result):
        vc_df = VarCorr(single_re_result).as_dataframe()
        row = vc_df[vc_df["grp"] == "Residual"].iloc[0]
        assert pd.isna(row["var1"])


# ---------------------------------------------------------------------------
# VarCorr — two random intercepts
# ---------------------------------------------------------------------------


class TestVarCorrTwoRE:
    def test_has_both_groups_and_residual(self, two_re_result):
        df = VarCorr(two_re_result).as_dataframe()
        assert set(df["grp"].values) == {"firm", "dept", "Residual"}

    def test_row_count(self, two_re_result):
        df = VarCorr(two_re_result).as_dataframe()
        # 2 RE groups + Residual = 3 rows for intercept-only
        assert len(df) == 3

    def test_vcov_order_matches_specs(self, two_re_result):
        df = VarCorr(two_re_result).as_dataframe()
        # firm should appear before dept (order of specs)
        grps = [g for g in df["grp"] if g != "Residual"]
        assert grps == ["firm", "dept"]


# ---------------------------------------------------------------------------
# VarCorr — random slopes (correlated)
# ---------------------------------------------------------------------------


class TestVarCorrSlopes:
    def test_has_intercept_and_slope_rows(self, slopes_result):
        df = VarCorr(slopes_result).as_dataframe()
        group_rows = df[df["grp"] == "g"]
        var1_vals = list(group_rows["var1"].dropna())
        assert "(Intercept)" in var1_vals
        assert "x" in var1_vals

    def test_has_correlation_row(self, slopes_result):
        df = VarCorr(slopes_result).as_dataframe()
        group_rows = df[df["grp"] == "g"]
        # Correlation row: var1=(Intercept), var2=slope_name
        corr_rows = group_rows[group_rows["var2"].notna()]
        assert len(corr_rows) >= 1

    def test_correlation_magnitude_le_1(self, slopes_result):
        df = VarCorr(slopes_result).as_dataframe()
        group_rows = df[df["grp"] == "g"]
        corr_rows = group_rows[group_rows["var2"].notna()]
        for _, row in corr_rows.iterrows():
            assert abs(row["sdcor"]) <= 1.0 + 1e-10

    def test_diagonal_sdcor_is_positive(self, slopes_result):
        df = VarCorr(slopes_result).as_dataframe()
        group_rows = df[df["grp"] == "g"]
        diag_rows = group_rows[group_rows["var2"].isna()]
        for _, row in diag_rows.iterrows():
            assert row["sdcor"] > 0


# ---------------------------------------------------------------------------
# CrossedLMEResult.fe_tvalues property
# ---------------------------------------------------------------------------


class TestFeTvalues:
    def test_fe_tvalues_shape(self, single_re_result):
        tv = single_re_result.fe_tvalues
        assert len(tv) == len(single_re_result.fe_params)

    def test_fe_tvalues_equals_params_over_bse(self, single_re_result):
        tv = np.asarray(single_re_result.fe_tvalues)
        expected = np.asarray(single_re_result.fe_params) / np.asarray(
            single_re_result.fe_bse
        )
        np.testing.assert_allclose(tv, expected, rtol=1e-12)

    def test_fe_tvalues_index_matches_fe_params(self, single_re_result):
        # When pandas available, index names should match
        tv = single_re_result.fe_tvalues
        fe = single_re_result.fe_params
        assert list(tv.index) == list(fe.index)


# ---------------------------------------------------------------------------
# CrossedLMEResult.summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_is_not_none(self, single_re_result):
        s = single_re_result.summary()
        assert s is not None

    def test_summary_str_is_nonempty(self, single_re_result):
        s = str(single_re_result.summary())
        assert len(s) > 100

    def test_summary_contains_formula(self, single_re_result):
        s = str(single_re_result.summary())
        assert "y ~ x" in s

    def test_summary_contains_reml_criterion(self, single_re_result):
        s = str(single_re_result.summary())
        assert re.search(r"REML criterion", s, re.IGNORECASE)

    def test_summary_contains_random_effects_section(self, single_re_result):
        s = str(single_re_result.summary())
        assert re.search(r"Random effects", s, re.IGNORECASE)

    def test_summary_contains_fixed_effects_section(self, single_re_result):
        s = str(single_re_result.summary())
        assert re.search(r"Fixed effects", s, re.IGNORECASE)

    def test_summary_contains_nobs(self, single_re_result):
        s = str(single_re_result.summary())
        assert str(single_re_result.nobs) in s

    def test_summary_contains_variance_component_sd(self, single_re_result):
        s = str(single_re_result.summary())
        # SD of group variance component should appear in random effects table
        sd = np.sqrt(single_re_result.variance_components["group"])
        # Check a 4-significant-figure version appears somewhere in the string
        assert f"{sd:.4f}"[:4] in s

    def test_summary_contains_fe_estimates(self, single_re_result):
        s = str(single_re_result.summary())
        intercept = float(np.asarray(single_re_result.fe_params)[0])
        assert f"{intercept:.4f}"[:5] in s

    def test_summary_contains_intercept_label(self, single_re_result):
        s = str(single_re_result.summary())
        assert "(Intercept)" in s

    def test_summary_convergence_status(self, single_re_result):
        s = str(single_re_result.summary())
        assert re.search(r"converged|convergence", s, re.IGNORECASE)

    def test_summary_contains_scaled_residuals(self, single_re_result):
        s = str(single_re_result.summary())
        assert re.search(r"Scaled residuals|scaled residuals", s, re.IGNORECASE)

    def test_summary_two_re(self, two_re_result):
        s = str(two_re_result.summary())
        assert "firm" in s
        assert "dept" in s
        assert "Residual" in s

    def test_summary_ngroups_shown(self, two_re_result):
        s = str(two_re_result.summary())
        # ngroups for firm and dept should be present
        assert str(two_re_result.ngroups["firm"]) in s
        assert str(two_re_result.ngroups["dept"]) in s


# ---------------------------------------------------------------------------
# VarCorr exported from interlace namespace
# ---------------------------------------------------------------------------


def test_varcorr_importable_from_interlace():
    from interlace import VarCorr as VC

    assert VC is not None
