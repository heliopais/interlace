"""Tests for emmeans() — estimated marginal means for LMMs.

Acceptance criteria:
  - emmeans(model, specs) returns a DataFrame with columns:
    <specs>, estimate, SE, df, lower, upper, t.ratio, p.value
  - One row per level of the specs factor (or per level combination)
  - SE > 0, df > 0, lower <= estimate <= upper
  - Estimates respect true treatment-effect ordering
  - Approximate accuracy: treatment differences close to data-generating values
  - Works for a single categorical predictor
  - Works for two crossed categorical predictors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

import interlace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_way_result():
    """y ~ trt + x + (1|group); trt has 3 levels with known effects."""
    rng = np.random.default_rng(42)
    n_groups = 20
    n_per_cell = 10
    n_trt = 3
    n = n_groups * n_per_cell * n_trt

    group_ids = np.repeat(np.arange(n_groups), n_per_cell * n_trt)
    trt = np.tile(np.repeat(["A", "B", "C"], n_per_cell), n_groups)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, n_groups)
    trt_effects = {"A": 0.0, "B": 2.0, "C": 4.0}
    y = (
        10.0
        + np.array([trt_effects[t] for t in trt])
        + 0.5 * x
        + u[group_ids]
        + rng.normal(0, 0.3, n)
    )
    df = pd.DataFrame({"y": y, "trt": trt, "x": x, "group": group_ids.astype(str)})
    return interlace.fit("y ~ trt + x", data=df, groups="group")


@pytest.fixture(scope="module")
def two_way_result():
    """y ~ A + B + (1|group); A has 2 levels, B has 3 levels."""
    rng = np.random.default_rng(99)
    n_groups = 15
    n_per_cell = 8
    levels_a = ["X", "Y"]
    levels_b = ["p", "q", "r"]
    n_a, n_b = len(levels_a), len(levels_b)
    n = n_groups * n_per_cell * n_a * n_b

    group_ids = np.repeat(np.arange(n_groups), n_per_cell * n_a * n_b)
    a_col = np.tile(np.repeat(levels_a, n_per_cell * n_b), n_groups)
    b_col = np.tile(np.tile(np.repeat(levels_b, n_per_cell), n_a), n_groups)
    u = rng.normal(0, 0.8, n_groups)
    a_eff = {"X": 0.0, "Y": 1.5}
    b_eff = {"p": 0.0, "q": 1.0, "r": 2.0}
    y = (
        5.0
        + np.array([a_eff[a] for a in a_col])
        + np.array([b_eff[b] for b in b_col])
        + u[group_ids]
        + rng.normal(0, 0.4, n)
    )
    df = pd.DataFrame({"y": y, "A": a_col, "B": b_col, "group": group_ids.astype(str)})
    return interlace.fit("y ~ A + B", data=df, groups="group")


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestEmmeansStructure:
    def test_returns_dataframe(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert isinstance(out, pd.DataFrame)

    def test_required_columns_present(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        for col in ["estimate", "SE", "df", "lower", "upper", "t.ratio", "p.value"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_one_row_per_level_single_factor(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert len(out) == 3

    def test_specs_column_in_output(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert "trt" in out.columns

    def test_all_levels_present(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert set(out["trt"]) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------


class TestEmmeansValidity:
    def test_se_positive(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert (out["SE"] > 0).all(), f"Non-positive SEs: {out['SE'].tolist()}"

    def test_df_positive(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert (out["df"] > 0).all(), f"Non-positive DFs: {out['df'].tolist()}"

    def test_df_finite(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert np.all(np.isfinite(out["df"])), f"Non-finite DFs: {out['df'].tolist()}"

    def test_ci_contains_estimate(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert (out["lower"] <= out["estimate"]).all()
        assert (out["estimate"] <= out["upper"]).all()

    def test_t_ratio_equals_estimate_over_se(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        expected = out["estimate"] / out["SE"]
        np.testing.assert_allclose(out["t.ratio"], expected, rtol=1e-10)

    def test_pvalue_consistent_with_t_ratio_and_df(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        expected = 2.0 * (1.0 - stats.t.cdf(np.abs(out["t.ratio"]), df=out["df"]))
        np.testing.assert_allclose(out["p.value"], expected, rtol=1e-10)

    def test_pvalue_in_01(self, one_way_result):
        out = interlace.emmeans(one_way_result, "trt")
        assert (out["p.value"] >= 0).all() and (out["p.value"] <= 1).all()


# ---------------------------------------------------------------------------
# Approximate accuracy: estimates reflect data-generating process
# ---------------------------------------------------------------------------


class TestEmmeansAccuracy:
    def test_treatment_ordering(self, one_way_result):
        """A < B < C (true effects 0, 2, 4)."""
        out = interlace.emmeans(one_way_result, "trt").set_index("trt")
        assert out.loc["A", "estimate"] < out.loc["B", "estimate"]
        assert out.loc["B", "estimate"] < out.loc["C", "estimate"]

    def test_treatment_differences_close_to_truth(self, one_way_result):
        """B - A ≈ 2.0, C - A ≈ 4.0 (within 0.5 units)."""
        out = interlace.emmeans(one_way_result, "trt").set_index("trt")
        diff_ba = out.loc["B", "estimate"] - out.loc["A", "estimate"]
        diff_ca = out.loc["C", "estimate"] - out.loc["A", "estimate"]
        assert abs(diff_ba - 2.0) < 0.5, f"B-A = {diff_ba:.3f}, expected ≈ 2.0"
        assert abs(diff_ca - 4.0) < 0.5, f"C-A = {diff_ca:.3f}, expected ≈ 4.0"

    def test_estimates_near_grand_mean_plus_effect(self, one_way_result):
        """Estimates centred around the overall intercept (~10)."""
        out = interlace.emmeans(one_way_result, "trt")
        overall_mean = out["estimate"].mean()
        # True: 10 + (0+2+4)/3 = 12
        assert 9.0 < overall_mean < 15.0, f"Overall mean = {overall_mean:.2f}"


# ---------------------------------------------------------------------------
# Two-factor marginalisation
# ---------------------------------------------------------------------------


class TestEmmeansTwoFactor:
    def test_two_factor_row_count(self, two_way_result):
        out = interlace.emmeans(two_way_result, ["A", "B"])
        assert len(out) == 6  # 2 × 3

    def test_two_factor_columns_present(self, two_way_result):
        out = interlace.emmeans(two_way_result, ["A", "B"])
        assert "A" in out.columns and "B" in out.columns

    def test_two_factor_all_combinations(self, two_way_result):
        out = interlace.emmeans(two_way_result, ["A", "B"])
        combos = set(zip(out["A"], out["B"], strict=True))
        expected = {
            ("X", "p"),
            ("X", "q"),
            ("X", "r"),
            ("Y", "p"),
            ("Y", "q"),
            ("Y", "r"),
        }
        assert combos == expected

    def test_two_factor_b_ordering(self, two_way_result):
        """Marginalising over A: p < q < r."""
        out = interlace.emmeans(two_way_result, "B").set_index("B")
        assert out.loc["p", "estimate"] < out.loc["q", "estimate"]
        assert out.loc["q", "estimate"] < out.loc["r", "estimate"]


# ---------------------------------------------------------------------------
# Custom `at` values
# ---------------------------------------------------------------------------


class TestEmmeansAt:
    def test_at_changes_estimate(self, one_way_result):
        """Setting x to a non-mean value shifts estimates by beta_x * delta."""
        base = interlace.emmeans(one_way_result, "trt")
        at_high = interlace.emmeans(one_way_result, "trt", at={"x": 5.0})
        # All estimates should shift in the same direction as x*beta_x
        # (beta_x ≈ 0.5, so shift should be about 0.5 * (5 - mean_x) ≈ 2.5)
        shifts = (
            at_high.set_index("trt")["estimate"] - base.set_index("trt")["estimate"]
        )
        # All shifts should have the same sign
        assert (shifts > 0).all() or (shifts < 0).all(), (
            f"Shifts not consistent: {shifts.tolist()}"
        )
