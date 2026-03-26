"""Tests for result.py (CrossedLMEResult, ModelInfo) and fit() entry point."""

import numpy as np
import pandas as pd
import pytest

from interlace import fit
from interlace.result import CrossedLMEResult

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n, q = 120, 6
    group_codes = np.tile(np.arange(q), n // q)
    b = rng.normal(scale=1.0, size=q)
    x1 = rng.normal(size=n)
    y = 2.0 + 0.5 * x1 + b[group_codes] + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "group": group_codes.astype(str)})


@pytest.fixture()
def crossed_df() -> pd.DataFrame:
    """Two crossed RE factors: 8 firms x 4 depts, 160 obs."""
    rng = np.random.default_rng(13)
    n_firm, n_dept = 8, 4
    n = 160
    firm = np.tile(np.arange(n_firm), n // n_firm)
    dept = np.repeat(np.arange(n_dept), n // n_dept)
    b_firm = rng.normal(scale=1.2, size=n_firm)
    b_dept = rng.normal(scale=0.8, size=n_dept)
    x1 = rng.normal(size=n)
    y = 1.0 + 0.7 * x1 + b_firm[firm] + b_dept[dept] + rng.normal(scale=1.0, size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "firm": firm.astype(str),
            "dept": dept.astype(str),
        }
    )


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_has_required_attrs(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        m = result.model
        assert hasattr(m, "exog")
        assert hasattr(m, "endog")
        assert hasattr(m, "groups")
        assert hasattr(m, "endog_names")
        assert hasattr(m, "formula")

    def test_data_frame_attr(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.model.data.frame is simple_df

    def test_exog_shape(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.model.exog.shape == (120, 2)  # intercept + x1

    def test_endog_names(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.model.endog_names == "y"

    def test_formula_stored(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.model.formula == "y ~ x1"


# ---------------------------------------------------------------------------
# CrossedLMEResult — structure
# ---------------------------------------------------------------------------


class TestCrossedLMEResultStructure:
    def test_returns_crossed_lme_result(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert isinstance(result, CrossedLMEResult)

    def test_converged(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.converged

    def test_fe_params_is_series(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert isinstance(result.fe_params, pd.Series)
        assert "Intercept" in result.fe_params.index
        assert "x1" in result.fe_params.index

    def test_fe_bse_shape(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.fe_bse.shape == result.fe_params.shape

    def test_fe_pvalues_in_01(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert (result.fe_pvalues >= 0).all()
        assert (result.fe_pvalues <= 1).all()

    def test_fe_conf_int_shape(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.fe_conf_int.shape == (2, 2)  # 2 params, lower/upper

    def test_resid_shape(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.resid.shape == (120,)

    def test_fittedvalues_shape(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.fittedvalues.shape == (120,)

    def test_resid_plus_fitted_equals_y(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        np.testing.assert_allclose(
            result.resid + result.fittedvalues,
            simple_df["y"].values,
            rtol=1e-10,
        )

    def test_scale_positive(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.scale > 0

    def test_random_effects_keys(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert "group" in result.random_effects
        re = result.random_effects["group"]
        assert isinstance(re, pd.Series)
        assert len(re) == 6  # 6 unique group levels

    def test_variance_components(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert "group" in result.variance_components
        assert result.variance_components["group"] > 0

    def test_nobs(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.nobs == 120

    def test_ngroups(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result.ngroups["group"] == 6


# ---------------------------------------------------------------------------
# CrossedLMEResult — statistical correctness
# ---------------------------------------------------------------------------


class TestCrossedLMEResultStats:
    def test_fe_params_close_to_truth(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert abs(result.fe_params["Intercept"] - 2.0) < 0.5
        assert abs(result.fe_params["x1"] - 0.5) < 0.3

    def test_matches_statsmodels_fe(self, simple_df: pd.DataFrame) -> None:
        import statsmodels.formula.api as smf

        result = fit("y ~ x1", simple_df, groups="group")
        sm = smf.mixedlm("y ~ x1", simple_df, groups=simple_df["group"]).fit(
            reml=True, method="lbfgs"
        )

        if abs(sm.fe_params["Intercept"]) < 0.1:
            pytest.skip("statsmodels converged to degenerate solution on this platform")

        np.testing.assert_allclose(
            result.fe_params.values,
            [sm.fe_params["Intercept"], sm.fe_params["x1"]],
            rtol=1e-2,
        )
        assert abs(result.scale - sm.scale) / sm.scale < 0.02

    def test_blups_shrink_toward_zero(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        re = result.random_effects["group"]
        # BLUPs should have smaller variance than raw group means
        raw_means = (
            simple_df.groupby("group")["y"].mean() - result.fe_params["Intercept"]
        )
        assert re.std() < raw_means.std()


# ---------------------------------------------------------------------------
# Crossed RE (two factors)
# ---------------------------------------------------------------------------


class TestCrossedRE:
    def test_two_re_groups(self, crossed_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", crossed_df, groups=["firm", "dept"])
        assert "firm" in result.random_effects
        assert "dept" in result.random_effects
        assert len(result.random_effects["firm"]) == 8
        assert len(result.random_effects["dept"]) == 4

    def test_two_re_variance_components(self, crossed_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", crossed_df, groups=["firm", "dept"])
        assert result.variance_components["firm"] > 0
        assert result.variance_components["dept"] > 0

    def test_two_re_resid_plus_fitted(self, crossed_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", crossed_df, groups=["firm", "dept"])
        np.testing.assert_allclose(
            result.resid + result.fittedvalues,
            crossed_df["y"].values,
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# statsmodels compatibility attributes (h0j)
# ---------------------------------------------------------------------------


class TestStatsmodelsCompat:
    def test_gpgap_group_col(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result._gpgap_group_col == "group"

    def test_gpgap_vc_cols_single(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        assert result._gpgap_vc_cols == []

    def test_gpgap_vc_cols_crossed(self, crossed_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", crossed_df, groups=["firm", "dept"])
        assert result._gpgap_group_col == "firm"
        assert result._gpgap_vc_cols == ["dept"]

    def test_model_groups_attr(self, simple_df: pd.DataFrame) -> None:
        result = fit("y ~ x1", simple_df, groups="group")
        np.testing.assert_array_equal(result.model.groups, simple_df["group"].values)


# ---------------------------------------------------------------------------
# Random slopes — random_effects and variance_components structure (hnl)
# ---------------------------------------------------------------------------


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


class TestRandomSlopesResult:
    def test_random_effects_is_dataframe_for_slopes(
        self, slope_df: pd.DataFrame
    ) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x | g)"])
        re = result.random_effects["g"]
        assert isinstance(re, pd.DataFrame)

    def test_random_effects_dataframe_index_is_group_levels(
        self, slope_df: pd.DataFrame
    ) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x | g)"])
        re = result.random_effects["g"]
        expected_levels = sorted(slope_df["g"].unique())
        assert list(re.index) == expected_levels

    def test_random_effects_dataframe_columns_are_term_names(
        self, slope_df: pd.DataFrame
    ) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x | g)"])
        re = result.random_effects["g"]
        assert list(re.columns) == ["(Intercept)", "x"]

    def test_random_effects_dataframe_shape(self, slope_df: pd.DataFrame) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x | g)"])
        re = result.random_effects["g"]
        assert re.shape == (10, 2)  # 10 groups, 2 terms

    def test_variance_components_is_dataframe_for_slopes(
        self, slope_df: pd.DataFrame
    ) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x | g)"])
        vc = result.variance_components["g"]
        assert isinstance(vc, pd.DataFrame)

    def test_variance_components_dataframe_shape(self, slope_df: pd.DataFrame) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x | g)"])
        vc = result.variance_components["g"]
        assert vc.shape == (2, 2)

    def test_variance_components_dataframe_labels(self, slope_df: pd.DataFrame) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x | g)"])
        vc = result.variance_components["g"]
        assert list(vc.columns) == ["(Intercept)", "x"]
        assert list(vc.index) == ["(Intercept)", "x"]

    def test_variance_components_is_positive_semidefinite(
        self, slope_df: pd.DataFrame
    ) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x | g)"])
        vc = result.variance_components["g"]
        eigenvalues = np.linalg.eigvalsh(vc.values)
        assert np.all(eigenvalues >= -1e-10)

    def test_independent_slopes_vc_is_diagonal(self, slope_df: pd.DataFrame) -> None:
        result = fit("y ~ x", slope_df, random=["(1 + x || g)"])
        vc = result.variance_components["g"]
        assert isinstance(vc, pd.DataFrame)
        # Off-diagonal entries should be zero for independent parameterisation
        off_diag = vc.values[~np.eye(2, dtype=bool)]
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)

    def test_intercept_only_random_effects_still_series(
        self, simple_df: pd.DataFrame
    ) -> None:
        """Backward compat: intercept-only groups remain pd.Series."""
        result = fit("y ~ x1", simple_df, groups="group")
        assert isinstance(result.random_effects["group"], pd.Series)

    def test_intercept_only_variance_components_still_scalar(
        self, simple_df: pd.DataFrame
    ) -> None:
        """Backward compat: intercept-only variance components remain float."""
        result = fit("y ~ x1", simple_df, groups="group")
        assert isinstance(result.variance_components["group"], float)
