"""Smoke tests for fit() with the new random= parameter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace


@pytest.fixture()
def slope_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n, q = 200, 10
    group_codes = np.repeat(np.arange(q), n // q)
    x = rng.normal(size=n)
    b_int = rng.normal(scale=0.8, size=q)
    b_slope = rng.normal(scale=0.4, size=q)
    y = 1.0 + 0.5 * x + b_int[group_codes] + b_slope[group_codes] * x
    y += rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": group_codes.astype(str)})


class TestFitRandomParameter:
    def test_accepts_random_param(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        assert result is not None

    def test_converged_correlated(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        assert result.converged

    def test_finite_llf_correlated(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        assert np.isfinite(result.llf)

    def test_converged_independent(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x || g)"])
        assert result.converged

    def test_finite_llf_independent(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x || g)"])
        assert np.isfinite(result.llf)

    def test_correlated_and_independent_give_different_llf(
        self, slope_df: pd.DataFrame
    ) -> None:
        r_corr = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        r_indep = interlace.fit("y ~ x", data=slope_df, random=["(1 + x || g)"])
        assert abs(r_corr.llf - r_indep.llf) > 1e-6

    def test_fe_params_shape(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        assert len(result.fe_params) == 2  # intercept + x

    def test_resid_length(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        assert len(result.resid) == len(slope_df)

    def test_fittedvalues_length(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        assert len(result.fittedvalues) == len(slope_df)


class TestFitBackwardCompat:
    def test_groups_param_unchanged(self, slope_df: pd.DataFrame) -> None:
        """groups= gives the same result as random=["(1 | g)"]."""
        r_old = interlace.fit("y ~ x", data=slope_df, groups="g")
        r_new = interlace.fit("y ~ x", data=slope_df, random=["(1 | g)"])
        np.testing.assert_allclose(
            r_old.fe_params.values, r_new.fe_params.values, rtol=1e-5
        )
        assert abs(r_old.scale - r_new.scale) / r_old.scale < 1e-5

    def test_groups_list_unchanged(self, slope_df: pd.DataFrame) -> None:
        df2 = slope_df.copy()
        df2["g2"] = np.tile(np.arange(5), len(df2) // 5).astype(str)
        r_old = interlace.fit("y ~ x", data=df2, groups=["g", "g2"])
        r_new = interlace.fit("y ~ x", data=df2, random=["(1 | g)", "(1 | g2)"])
        np.testing.assert_allclose(
            r_old.fe_params.values, r_new.fe_params.values, rtol=1e-5
        )
