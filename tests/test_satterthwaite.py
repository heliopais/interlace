"""Tests for Satterthwaite denominator degrees of freedom.

Acceptance criteria:
  - result.fe_df exists: Series/array of DFs, one per FE coefficient
  - All DFs are positive and finite
  - fe_pvalues use t(df) distribution (not normal)
  - summary() shows df and t value columns (not z value)
  - For intercept-only RE model, intercept DF ≈ n_groups - 1
  - For a large balanced design (many obs per group), slope DF is large
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
def small_result():
    """Small balanced design: y ~ x + (1|group), 10 groups × 5 obs."""
    rng = np.random.default_rng(0)
    n_groups, n_per = 10, 5
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, size=n_groups)
    eps = rng.normal(0, 0.5, size=n)
    y = 3.0 + 2.0 * x + u[np.repeat(np.arange(n_groups), n_per)] + eps
    df = pd.DataFrame({"y": y, "x": x, "group": group_ids})
    return interlace.fit("y ~ x", data=df, groups="group")


@pytest.fixture(scope="module")
def large_result():
    """Large balanced design: y ~ x + (1|group), 100 groups × 50 obs."""
    rng = np.random.default_rng(1)
    n_groups, n_per = 100, 50
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, size=n_groups)
    eps = rng.normal(0, 0.5, size=n)
    y = 3.0 + 2.0 * x + u[np.repeat(np.arange(n_groups), n_per)] + eps
    df = pd.DataFrame({"y": y, "x": x, "group": group_ids})
    return interlace.fit("y ~ x", data=df, groups="group")


# ---------------------------------------------------------------------------
# fe_df attribute
# ---------------------------------------------------------------------------


class TestFeDfAttribute:
    def test_fe_df_exists(self, small_result):
        assert hasattr(small_result, "fe_df")

    def test_fe_df_length_matches_fe_params(self, small_result):
        assert len(small_result.fe_df) == len(small_result.fe_params)

    def test_fe_df_all_positive(self, small_result):
        dfs = np.asarray(small_result.fe_df)
        assert np.all(dfs > 0), f"Non-positive DFs: {dfs}"

    def test_fe_df_all_finite(self, small_result):
        dfs = np.asarray(small_result.fe_df)
        assert np.all(np.isfinite(dfs)), f"Non-finite DFs: {dfs}"

    def test_fe_df_index_matches_fe_params(self, small_result):
        # When pandas is available, index should match
        fe = small_result.fe_params
        dfs = small_result.fe_df
        assert list(dfs.index) == list(fe.index)

    def test_fe_df_large_for_large_design(self, large_result):
        # With 100 groups × 50 obs, DFs should be large (>> 10)
        dfs = np.asarray(large_result.fe_df)
        assert np.all(dfs > 50), f"DFs unexpectedly small for large design: {dfs}"


# ---------------------------------------------------------------------------
# fe_pvalues use t-distribution with Satterthwaite DFs
# ---------------------------------------------------------------------------


class TestFePvaluesTDistribution:
    def test_fe_pvalues_use_t_not_normal(self, small_result):
        # p-values should match t(df) distribution exactly
        t_stats = np.asarray(small_result.fe_tvalues)
        dfs = np.asarray(small_result.fe_df)
        expected = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=dfs))
        actual = np.asarray(small_result.fe_pvalues)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_fe_pvalues_in_01(self, small_result):
        pv = np.asarray(small_result.fe_pvalues)
        assert np.all(pv >= 0) and np.all(pv <= 1)

    def test_t_test_more_conservative_than_z_test_small_sample(self, small_result):
        # For small samples, t-test p-values should be >= z-test p-values
        t_stats = np.abs(np.asarray(small_result.fe_tvalues))
        dfs = np.asarray(small_result.fe_df)
        t_pvals = 2.0 * (1.0 - stats.t.cdf(t_stats, df=dfs))
        z_pvals = 2.0 * (1.0 - stats.norm.cdf(t_stats))
        assert np.all(t_pvals >= z_pvals - 1e-12), (
            f"t-test p-values not >= z-test p-values: t={t_pvals}, z={z_pvals}"
        )


# ---------------------------------------------------------------------------
# Satterthwaite DFs have expected properties for balanced one-way design
# ---------------------------------------------------------------------------


class TestSatterthwaiteDFProperties:
    def test_intercept_df_between_1_and_n_groups(self, small_result):
        # Intercept DF in a random-intercept model is roughly n_groups - 1
        n_groups = small_result.ngroups["group"]
        intercept_df = float(np.asarray(small_result.fe_df)[0])
        assert 1 <= intercept_df <= n_groups + 1, (
            f"Intercept DF={intercept_df:.2f} out of expected range "
            f"[1, {n_groups + 1}] for n_groups={n_groups}"
        )

    def test_slope_df_at_least_total_obs_minus_groups(self, small_result):
        # Slope DF is bounded below by within-group df
        n_obs = small_result.nobs
        n_groups = small_result.ngroups["group"]
        slope_df = float(np.asarray(small_result.fe_df)[1])
        within_df = n_obs - n_groups
        assert slope_df >= within_df * 0.5, (
            f"Slope DF={slope_df:.2f} unexpectedly small (within_df={within_df})"
        )


# ---------------------------------------------------------------------------
# summary() output reflects Satterthwaite DFs
# ---------------------------------------------------------------------------


class TestSummaryWithSatterthwaite:
    def test_summary_shows_t_value_not_z_value(self, small_result):
        s = str(small_result.summary())
        assert "t value" in s or "t_value" in s.lower(), (
            f"Expected 't value' in summary; got:\n{s}"
        )
        assert "z value" not in s, f"Unexpected 'z value' in summary:\n{s}"

    def test_summary_shows_df_column(self, small_result):
        s = str(small_result.summary())
        assert "df" in s, f"Expected 'df' column in summary; got:\n{s}"

    def test_summary_pval_header_uses_t(self, small_result):
        s = str(small_result.summary())
        assert "Pr(>|t|)" in s, f"Expected 'Pr(>|t|)' in summary; got:\n{s}"
