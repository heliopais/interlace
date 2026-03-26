"""Phase 2 TDD: formula parsing with polars DataFrame input.

These tests verify that parse_formula() and extract_group_factors() produce
identical results whether given a pandas or polars DataFrame.
"""
# ruff: noqa: E402

import numpy as np
import pandas as pd
import pytest

polars = pytest.importorskip("polars")

from interlace.formula import extract_group_factors, parse_formula  # noqa: E402


@pytest.fixture()
def pdf() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = 30
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "group": np.tile(["a", "b", "c"], n // 3),
        }
    )


@pytest.fixture()
def pldf(pdf: pd.DataFrame) -> "polars.DataFrame":
    return polars.DataFrame(
        {
            "y": pdf["y"].to_numpy(),
            "x1": pdf["x1"].to_numpy(),
            "x2": pdf["x2"].to_numpy(),
            "group": pdf["group"].to_numpy(),
        }
    )


class TestParseFormulaPolars:
    def test_accepts_polars_dataframe(self, pldf: "polars.DataFrame") -> None:
        result = parse_formula("y ~ x1 + x2", data=pldf, groups="group")
        assert result is not None

    def test_X_matches_pandas(
        self, pdf: pd.DataFrame, pldf: "polars.DataFrame"
    ) -> None:
        pd_result = parse_formula("y ~ x1 + x2", data=pdf, groups="group")
        pl_result = parse_formula("y ~ x1 + x2", data=pldf, groups="group")
        np.testing.assert_allclose(pl_result.X, pd_result.X, atol=1e-12)

    def test_y_matches_pandas(
        self, pdf: pd.DataFrame, pldf: "polars.DataFrame"
    ) -> None:
        pd_result = parse_formula("y ~ x1 + x2", data=pdf, groups="group")
        pl_result = parse_formula("y ~ x1 + x2", data=pldf, groups="group")
        np.testing.assert_allclose(pl_result.y, pd_result.y, atol=1e-12)

    def test_groups_matches_pandas(
        self, pdf: pd.DataFrame, pldf: "polars.DataFrame"
    ) -> None:
        pd_result = parse_formula("y ~ x1 + x2", data=pdf, groups="group")
        pl_result = parse_formula("y ~ x1 + x2", data=pldf, groups="group")
        np.testing.assert_array_equal(pl_result.groups, pd_result.groups)

    def test_term_names_match_pandas(
        self, pdf: pd.DataFrame, pldf: "polars.DataFrame"
    ) -> None:
        pd_result = parse_formula("y ~ x1 + x2", data=pdf, groups="group")
        pl_result = parse_formula("y ~ x1 + x2", data=pldf, groups="group")
        assert pl_result.term_names == pd_result.term_names

    def test_no_intercept_polars(self, pldf: "polars.DataFrame") -> None:
        result = parse_formula("y ~ 0 + x1 + x2", data=pldf, groups="group")
        assert result.X.shape == (30, 2)
        assert "Intercept" not in result.term_names

    def test_missing_groups_column_polars(self, pldf: "polars.DataFrame") -> None:
        with pytest.raises(ValueError, match="groups"):
            parse_formula("y ~ x1", data=pldf, groups="nonexistent")


class TestExtractGroupFactorsPolars:
    def test_accepts_polars_dataframe(self, pldf: "polars.DataFrame") -> None:
        factors = extract_group_factors(pldf, ["group"])
        assert len(factors) == 1

    def test_codes_match_pandas(
        self, pdf: pd.DataFrame, pldf: "polars.DataFrame"
    ) -> None:
        pd_factors = extract_group_factors(pdf, ["group"])
        pl_factors = extract_group_factors(pldf, ["group"])
        np.testing.assert_array_equal(pl_factors[0][1], pd_factors[0][1])

    def test_n_levels_match_pandas(
        self, pdf: pd.DataFrame, pldf: "polars.DataFrame"
    ) -> None:
        pd_factors = extract_group_factors(pdf, ["group"])
        pl_factors = extract_group_factors(pldf, ["group"])
        assert pl_factors[0][2] == pd_factors[0][2]
