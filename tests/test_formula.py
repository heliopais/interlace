"""Tests for formula.py — fixed-effects formula parsing, statsmodels-style API."""
# ruff: noqa: E402

import numpy as np
import pandas as pd
import pytest

from interlace.formula import extract_group_factors, parse_formula


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "group": np.tile(["a", "b", "c"], n // 3),
        }
    )


class TestParseFormula:
    def test_returns_design_matrix_and_response(self, simple_df: pd.DataFrame) -> None:
        result = parse_formula("y ~ x1 + x2", data=simple_df, groups="group")
        assert hasattr(result, "X")
        assert hasattr(result, "y")
        assert hasattr(result, "groups")

    def test_design_matrix_shape(self, simple_df: pd.DataFrame) -> None:
        result = parse_formula("y ~ x1 + x2", data=simple_df, groups="group")
        # intercept + x1 + x2 = 3 columns
        assert result.X.shape == (60, 3)

    def test_design_matrix_has_intercept(self, simple_df: pd.DataFrame) -> None:
        result = parse_formula("y ~ x1 + x2", data=simple_df, groups="group")
        assert "Intercept" in result.term_names

    def test_response_vector(self, simple_df: pd.DataFrame) -> None:
        result = parse_formula("y ~ x1 + x2", data=simple_df, groups="group")
        np.testing.assert_array_equal(result.y, simple_df["y"].values)

    def test_groups_array(self, simple_df: pd.DataFrame) -> None:
        result = parse_formula("y ~ x1 + x2", data=simple_df, groups="group")
        np.testing.assert_array_equal(result.groups, simple_df["group"].values)

    def test_no_intercept_formula(self, simple_df: pd.DataFrame) -> None:
        result = parse_formula("y ~ 0 + x1 + x2", data=simple_df, groups="group")
        assert result.X.shape == (60, 2)
        assert "Intercept" not in result.term_names

    def test_missing_groups_column_raises(self, simple_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="groups"):
            parse_formula("y ~ x1", data=simple_df, groups="nonexistent")

    def test_groups_as_array(self, simple_df: pd.DataFrame) -> None:
        groups_arr = simple_df["group"].values
        result = parse_formula("y ~ x1 + x2", data=simple_df, groups=groups_arr)
        np.testing.assert_array_equal(result.groups, groups_arr)


class TestExtractGroupFactors:
    def test_returns_list_of_tuples(self, simple_df: pd.DataFrame) -> None:
        factors = extract_group_factors(simple_df, ["group"])
        assert isinstance(factors, list)
        assert len(factors) == 1
        name, codes, n_levels = factors[0]
        assert name == "group"
        assert isinstance(codes, np.ndarray)
        assert isinstance(n_levels, int)

    def test_n_levels(self, simple_df: pd.DataFrame) -> None:
        _, _, n_levels = extract_group_factors(simple_df, ["group"])[0]
        assert n_levels == 3  # a, b, c

    def test_codes_are_integer(self, simple_df: pd.DataFrame) -> None:
        _, codes, _ = extract_group_factors(simple_df, ["group"])[0]
        assert np.issubdtype(codes.dtype, np.integer)

    def test_codes_range(self, simple_df: pd.DataFrame) -> None:
        _, codes, n_levels = extract_group_factors(simple_df, ["group"])[0]
        assert codes.min() >= 0
        assert codes.max() < n_levels

    def test_codes_length(self, simple_df: pd.DataFrame) -> None:
        _, codes, _ = extract_group_factors(simple_df, ["group"])[0]
        assert len(codes) == len(simple_df)

    def test_multiple_groups(self, simple_df: pd.DataFrame) -> None:
        simple_df = simple_df.copy()
        simple_df["group2"] = np.tile(["x", "y"], len(simple_df) // 2)
        factors = extract_group_factors(simple_df, ["group", "group2"])
        assert len(factors) == 2
        assert factors[0][0] == "group"
        assert factors[1][0] == "group2"
        assert factors[1][2] == 2  # x, y
