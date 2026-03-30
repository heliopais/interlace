"""Tests for Type II and Type III ANOVA F-tables for LMMs.

TDD: written before the implementation.

Acceptance criteria:
  - anova_type3(result) returns DataFrame with columns ["term","df1","df2","F","Pr(>F)"]
  - anova_type2(result) returns same column structure
  - Intercept is excluded from the table
  - df1 = number of columns belonging to the term
  - df2 > 0 and finite (Satterthwaite approximation)
  - F > 0 for a predictor that is clearly significant
  - p-values are in [0, 1]
  - For a single continuous predictor, F == t²  (Type III Wald identity)
  - Type II and Type III agree for additive models with a single numeric predictor
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace.anova_type import anova_type2, anova_type3

FIXTURES = Path(__file__).parent / "fixtures"

EXPECTED_COLS = ["term", "df1", "df2", "F", "Pr(>F)"]


@pytest.fixture(scope="module")
def two_re_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "two_re_data.csv")


@pytest.fixture(scope="module")
def reml_result(two_re_data):
    return interlace.fit(
        "y ~ x", data=two_re_data, groups=["firm", "dept"], method="REML"
    )


@pytest.fixture(scope="module")
def ml_result(two_re_data):
    return interlace.fit(
        "y ~ x", data=two_re_data, groups=["firm", "dept"], method="ML"
    )


# ---------------------------------------------------------------------------
# anova_type3(): structure
# ---------------------------------------------------------------------------


def test_type3_returns_dataframe(reml_result):
    tbl = anova_type3(reml_result)
    assert isinstance(tbl, pd.DataFrame)


def test_type3_has_correct_columns(reml_result):
    tbl = anova_type3(reml_result)
    for col in EXPECTED_COLS:
        assert col in tbl.columns, f"Missing column: {col!r}"


def test_type3_excludes_intercept(reml_result):
    tbl = anova_type3(reml_result)
    assert "Intercept" not in tbl["term"].values
    assert "(Intercept)" not in tbl["term"].values


def test_type3_one_row_per_non_intercept_term(reml_result):
    tbl = anova_type3(reml_result)
    # "y ~ x" has one non-intercept term: x
    assert len(tbl) == 1
    assert tbl["term"].iloc[0] == "x"


def test_type3_df1_is_one_for_continuous(reml_result):
    tbl = anova_type3(reml_result)
    assert tbl["df1"].iloc[0] == 1


def test_type3_df2_positive_finite(reml_result):
    tbl = anova_type3(reml_result)
    df2 = tbl["df2"].iloc[0]
    assert np.isfinite(df2)
    assert df2 > 0


def test_type3_f_positive(reml_result):
    tbl = anova_type3(reml_result)
    assert tbl["F"].iloc[0] > 0


def test_type3_pvalue_in_range(reml_result):
    tbl = anova_type3(reml_result)
    p = tbl["Pr(>F)"].iloc[0]
    assert 0.0 <= p <= 1.0


def test_type3_f_equals_t_squared_for_single_term(reml_result):
    """For df1=1, the Wald F statistic must equal the squared t-statistic."""
    tbl = anova_type3(reml_result)
    F_val = float(tbl["F"].iloc[0])
    t_val = float(reml_result.fe_params["x"] / reml_result.fe_bse["x"])
    assert abs(F_val - t_val**2) < 1e-6, f"F={F_val:.6f}, t²={t_val**2:.6f}"


def test_type3_df2_matches_satterthwaite(reml_result):
    """df2 for a single-column term must equal the Satterthwaite DF for that coeff."""
    from interlace.satterthwaite import satterthwaite_dfs

    tbl = anova_type3(reml_result)
    # fe_params index: ["Intercept", "x"] — x is index 1
    sat_dfs = satterthwaite_dfs(reml_result)
    x_idx = list(reml_result.fe_params.index).index("x")
    expected_df2 = sat_dfs[x_idx]
    actual_df2 = float(tbl["df2"].iloc[0])
    assert abs(actual_df2 - expected_df2) < 1e-6, (
        f"df2={actual_df2:.4f}, Satterthwaite={expected_df2:.4f}"
    )


# ---------------------------------------------------------------------------
# anova_type2(): structure
# ---------------------------------------------------------------------------


def test_type2_returns_dataframe(reml_result):
    tbl = anova_type2(reml_result)
    assert isinstance(tbl, pd.DataFrame)


def test_type2_has_correct_columns(reml_result):
    tbl = anova_type2(reml_result)
    for col in EXPECTED_COLS:
        assert col in tbl.columns, f"Missing column: {col!r}"


def test_type2_excludes_intercept(reml_result):
    tbl = anova_type2(reml_result)
    assert "Intercept" not in tbl["term"].values
    assert "(Intercept)" not in tbl["term"].values


def test_type2_one_row_per_non_intercept_term(reml_result):
    tbl = anova_type2(reml_result)
    assert len(tbl) == 1
    assert tbl["term"].iloc[0] == "x"


def test_type2_df1_is_one_for_continuous(reml_result):
    tbl = anova_type2(reml_result)
    assert tbl["df1"].iloc[0] == 1


def test_type2_df2_positive_finite(reml_result):
    tbl = anova_type2(reml_result)
    df2 = tbl["df2"].iloc[0]
    assert np.isfinite(df2)
    assert df2 > 0


def test_type2_f_positive(reml_result):
    tbl = anova_type2(reml_result)
    assert tbl["F"].iloc[0] > 0


def test_type2_pvalue_in_range(reml_result):
    tbl = anova_type2(reml_result)
    p = tbl["Pr(>F)"].iloc[0]
    assert 0.0 <= p <= 1.0


def test_type2_accepts_reml_result(reml_result):
    """anova_type2 accepts a REML-fitted model (refits as ML internally)."""
    tbl = anova_type2(reml_result)
    assert isinstance(tbl, pd.DataFrame)
    assert tbl["F"].iloc[0] > 0


def test_type2_accepts_ml_result(ml_result):
    """anova_type2 must work when given an ML-fitted model."""
    tbl = anova_type2(ml_result)
    assert isinstance(tbl, pd.DataFrame)
    assert tbl["F"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Type II vs III agreement for simple additive model
# ---------------------------------------------------------------------------


def test_type2_and_type3_f_close_for_single_numeric_term(reml_result):
    """For a single continuous predictor, Type II and Type III F must agree closely.

    Both reduce to testing β_x = 0 which has only one definition for a
    single-predictor additive model.  The two methods (Wald vs LRT-derived)
    may differ slightly due to the LRT vs Wald approximation, but should agree
    within 5 % on the F value.
    """
    t3 = anova_type3(reml_result)
    t2 = anova_type2(reml_result)
    f3 = float(t3["F"].iloc[0])
    f2 = float(t2["F"].iloc[0])
    # Relative tolerance of 5%
    assert abs(f2 - f3) / f3 < 0.05, f"Type II F={f2:.4f}, Type III F={f3:.4f}"


# ---------------------------------------------------------------------------
# R-parity fixtures (skipped if not generated)
# ---------------------------------------------------------------------------

ANOVA_TYPE_FIXTURE = FIXTURES / "anova_type_r_results.json"


@pytest.mark.skipif(
    not ANOVA_TYPE_FIXTURE.exists(),
    reason="R ANOVA fixture not generated; run tests/fixtures/gen_anova_type.R",
)
def test_type3_f_matches_lmertest(two_re_data):
    r = json.loads(ANOVA_TYPE_FIXTURE.read_text())
    result = interlace.fit(
        "y ~ x", data=two_re_data, groups=["firm", "dept"], method="REML"
    )
    tbl = anova_type3(result)
    r_f = r["type3"]["x"]["F"]
    il_f = float(tbl[tbl["term"] == "x"]["F"].iloc[0])
    assert abs(il_f - r_f) / r_f < 0.01, f"F: interlace={il_f:.4f}, R={r_f:.4f}"


@pytest.mark.skipif(
    not ANOVA_TYPE_FIXTURE.exists(),
    reason="R ANOVA fixture not generated; run tests/fixtures/gen_anova_type.R",
)
def test_type2_f_matches_car_anova(two_re_data):
    r = json.loads(ANOVA_TYPE_FIXTURE.read_text())
    result = interlace.fit(
        "y ~ x", data=two_re_data, groups=["firm", "dept"], method="REML"
    )
    tbl = anova_type2(result)
    r_f = r["type2"]["x"]["F"]
    il_f = float(tbl[tbl["term"] == "x"]["F"].iloc[0])
    assert abs(il_f - r_f) / r_f < 0.01, f"F: interlace={il_f:.4f}, R={r_f:.4f}"
