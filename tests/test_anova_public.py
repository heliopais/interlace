"""Tests for the public Anova() entry point.

TDD: written before implementation.

Acceptance criteria:
  - Anova(model) defaults to type='III', test='F'
  - Anova(model, type='II'|'III', test='F') → same DataFrame as anova_type2/3
  - Anova(model, test='Chisq') → columns ["term","df","Chisq","Pr(>Chisq)"]

  - Chisq = F * df1 for the Wald chi-square conversion
  - Anova([m1, m2]) → LRT comparison table (same as anova(m1, m2))
  - ValueError on unsupported type= or test= values
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import interlace
from interlace.anova_type import Anova, anova_type2, anova_type3

FIXTURES = Path(__file__).parent / "fixtures"


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
# Default behaviour: type='III', test='F'
# ---------------------------------------------------------------------------


def test_anova_default_returns_dataframe(reml_result):
    tbl = Anova(reml_result)
    assert isinstance(tbl, pd.DataFrame)


def test_anova_default_matches_type3_f(reml_result):
    """Anova(model) should produce exactly the same table as anova_type3(model)."""
    expected = anova_type3(reml_result)
    actual = Anova(reml_result)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_anova_type3_explicit_matches_type3(reml_result):
    expected = anova_type3(reml_result)
    actual = Anova(reml_result, type="III")
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_anova_type2_explicit_matches_type2(reml_result):
    expected = anova_type2(reml_result)
    actual = Anova(reml_result, type="II")
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# test='Chisq': Wald chi-square table
# ---------------------------------------------------------------------------

CHISQ_COLS = ["term", "df", "Chisq", "Pr(>Chisq)"]


def test_anova_chisq_returns_dataframe(reml_result):
    tbl = Anova(reml_result, test="Chisq")
    assert isinstance(tbl, pd.DataFrame)


def test_anova_chisq_has_correct_columns(reml_result):
    tbl = Anova(reml_result, test="Chisq")
    for col in CHISQ_COLS:
        assert col in tbl.columns, f"Missing column: {col!r}"


def test_anova_chisq_excludes_f_columns(reml_result):
    tbl = Anova(reml_result, test="Chisq")
    assert "F" not in tbl.columns
    assert "df2" not in tbl.columns


def test_anova_chisq_df_equals_df1(reml_result):
    """df in Chisq table must equal df1 from the F table."""
    f_tbl = anova_type3(reml_result)
    chisq_tbl = Anova(reml_result, test="Chisq")
    assert int(chisq_tbl["df"].iloc[0]) == int(f_tbl["df1"].iloc[0])


def test_anova_chisq_equals_f_times_df1(reml_result):
    """Wald Chisq = F * df1 (within floating-point precision)."""
    f_tbl = anova_type3(reml_result)
    chisq_tbl = Anova(reml_result, test="Chisq")
    expected_chisq = float(f_tbl["F"].iloc[0]) * int(f_tbl["df1"].iloc[0])
    actual_chisq = float(chisq_tbl["Chisq"].iloc[0])
    assert abs(actual_chisq - expected_chisq) < 1e-9, (
        f"Chisq={actual_chisq:.6f}, F*df1={expected_chisq:.6f}"
    )


def test_anova_chisq_pvalue_in_range(reml_result):
    tbl = Anova(reml_result, test="Chisq")
    p = tbl["Pr(>Chisq)"].iloc[0]
    assert 0.0 <= p <= 1.0


def test_anova_chisq_type2(reml_result):
    """test='Chisq' should also work with type='II'."""
    tbl = Anova(reml_result, type="II", test="Chisq")
    assert isinstance(tbl, pd.DataFrame)
    for col in CHISQ_COLS:
        assert col in tbl.columns


# ---------------------------------------------------------------------------
# List of models: LRT comparison
# ---------------------------------------------------------------------------


def test_anova_list_returns_dataframe(ml_result, two_re_data):
    ml_null = interlace.fit(
        "y ~ 1", data=two_re_data, groups=["firm", "dept"], method="ML"
    )
    tbl = Anova([ml_null, ml_result])
    assert isinstance(tbl, pd.DataFrame)


def test_anova_list_lrt_columns(ml_result, two_re_data):
    """LRT table should have the same columns as the existing anova() function."""
    from interlace.anova import anova as lrt_anova

    ml_null = interlace.fit(
        "y ~ 1", data=two_re_data, groups=["firm", "dept"], method="ML"
    )
    expected = lrt_anova(ml_null, ml_result)
    actual = Anova([ml_null, ml_result])
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_anova_list_reml_raises(reml_result, two_re_data):
    """LRT comparison of REML-fitted models must raise ValueError."""
    reml_null = interlace.fit(
        "y ~ 1", data=two_re_data, groups=["firm", "dept"], method="REML"
    )
    with pytest.raises(ValueError, match="REML"):
        Anova([reml_null, reml_result])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_anova_invalid_type_raises(reml_result):
    with pytest.raises(ValueError, match="type"):
        Anova(reml_result, type="IV")


def test_anova_invalid_test_raises(reml_result):
    with pytest.raises(ValueError, match="test"):
        Anova(reml_result, test="LRT")


# ---------------------------------------------------------------------------
# Public API: Anova exported from interlace top-level
# ---------------------------------------------------------------------------


def test_anova_exported_from_interlace():
    assert hasattr(interlace, "Anova")


def test_anova_callable_via_interlace(reml_result):
    tbl = interlace.Anova(reml_result)
    assert isinstance(tbl, pd.DataFrame)
