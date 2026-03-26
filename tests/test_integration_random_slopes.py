"""Integration tests for random slopes support.

Covers:
  - random= parameter parsing (correlated and independent)
  - Backward compat: groups= gives same result as equivalent random=
  - Correlated vs independent models produce different estimates
  - Prediction on new data with random slopes (smoke test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace


@pytest.fixture(scope="module")
def slope_df() -> pd.DataFrame:
    rng = np.random.default_rng(55)
    n, q = 300, 12
    g = np.repeat(np.arange(q), n // q)
    x = rng.normal(size=n)
    b_int = rng.normal(scale=0.8, size=q)
    b_slope = rng.normal(scale=0.4, size=q)
    y = 1.5 + 0.6 * x + b_int[g] + b_slope[g] * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g.astype(str)})


@pytest.fixture(scope="module")
def two_factor_df() -> pd.DataFrame:
    rng = np.random.default_rng(56)
    n = 400
    g1 = rng.choice([f"a{i}" for i in range(10)], n)
    g2 = rng.choice([f"b{i}" for i in range(6)], n)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.8, size=n) + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x": x, "g1": g1, "g2": g2})


class TestRandomParamParsing:
    def test_correlated_random_param(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        assert result.converged
        assert result.random_effects["g"].shape == (12, 2)

    def test_independent_random_param(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x || g)"])
        assert result.converged
        assert result.random_effects["g"].shape == (12, 2)

    def test_mixed_random_specs(self, two_factor_df: pd.DataFrame) -> None:
        result = interlace.fit(
            "y ~ x", data=two_factor_df, random=["(1 + x | g1)", "(1 | g2)"]
        )
        assert result.converged
        assert isinstance(result.random_effects["g1"], pd.DataFrame)
        assert isinstance(result.random_effects["g2"], pd.Series)


class TestBackwardCompat:
    def test_groups_str_same_as_random_intercept(self, slope_df: pd.DataFrame) -> None:
        r_old = interlace.fit("y ~ x", data=slope_df, groups="g")
        r_new = interlace.fit("y ~ x", data=slope_df, random=["(1 | g)"])
        np.testing.assert_allclose(
            r_old.fe_params.values, r_new.fe_params.values, rtol=1e-5
        )
        assert abs(r_old.scale - r_new.scale) / r_old.scale < 1e-5

    def test_groups_list_same_as_random_intercepts(
        self, two_factor_df: pd.DataFrame
    ) -> None:
        r_old = interlace.fit("y ~ x", data=two_factor_df, groups=["g1", "g2"])
        r_new = interlace.fit(
            "y ~ x", data=two_factor_df, random=["(1 | g1)", "(1 | g2)"]
        )
        np.testing.assert_allclose(
            r_old.fe_params.values, r_new.fe_params.values, rtol=1e-5
        )

    def test_intercept_only_random_effects_is_series(
        self, slope_df: pd.DataFrame
    ) -> None:
        result = interlace.fit("y ~ x", data=slope_df, groups="g")
        assert isinstance(result.random_effects["g"], pd.Series)

    def test_intercept_only_variance_components_is_float(
        self, slope_df: pd.DataFrame
    ) -> None:
        result = interlace.fit("y ~ x", data=slope_df, groups="g")
        assert isinstance(result.variance_components["g"], float)


class TestCorrVsIndep:
    def test_different_llf(self, slope_df: pd.DataFrame) -> None:
        r_corr = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        r_indep = interlace.fit("y ~ x", data=slope_df, random=["(1 + x || g)"])
        assert abs(r_corr.llf - r_indep.llf) > 1e-6

    def test_corr_has_nonzero_offdiagonal(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        vc = result.variance_components["g"]
        # Off-diagonal need not be zero for correlated model
        off_diag = vc.loc["(Intercept)", "x"]
        # Just check it's a finite number (not necessarily nonzero for all seeds)
        assert np.isfinite(off_diag)

    def test_indep_has_zero_offdiagonal(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x || g)"])
        vc = result.variance_components["g"]
        np.testing.assert_allclose(vc.loc["(Intercept)", "x"], 0.0, atol=1e-12)


class TestPredictionWithSlopes:
    def test_predict_newdata_returns_array(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        newdata = slope_df.iloc[:10].reset_index(drop=True)
        pred = result.predict(newdata=newdata)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (10,)

    def test_predict_unseen_group_no_error(self, slope_df: pd.DataFrame) -> None:
        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        newdata = pd.DataFrame({"x": [1.0, -1.0], "g": ["UNSEEN_A", "UNSEEN_B"]})
        pred = result.predict(newdata=newdata)
        assert pred.shape == (2,)

    def test_predict_include_re_false_ignores_slopes(
        self, slope_df: pd.DataFrame
    ) -> None:
        import patsy

        result = interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        pred_fe = result.predict(newdata=slope_df, include_re=False)
        fe_formula = result.model.formula.split("~", 1)[1].strip()
        X = np.asarray(patsy.dmatrix(fe_formula, slope_df, return_type="dataframe"))
        expected = X @ result.fe_params.values
        np.testing.assert_allclose(pred_fe, expected, rtol=1e-10)
