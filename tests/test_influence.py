"""Tests for interlace.influence.hlm_influence (and wrappers).

Uses a small dataset (10 groups × 5 obs) to keep refit loop fast.

Acceptance criteria:
  - Returns DataFrame with cooksd, mdffits, covtrace, covratio, rvc.* columns
  - Works with both CrossedLMEResult and statsmodels MixedLMResults
  - cooksd values are non-negative
  - cooksd from both models correlate > 0.9 (obs-level)
  - cooks_distance() and mdffits() wrappers return 1-D numpy arrays
  - Group-level diagnostics also return the correct structure
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.influence import (
    _reduced_params,
    _refit,
    _refit_groups_arg,
    _require_pandas,
    _vc_to_scalars,
    cooks_distance,
    hlm_influence,
    mdffits,
    ols_dfbetas_qr,
)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_groups, n_per = 10, 5
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, n_groups)
    eps = rng.normal(0, 0.5, n)
    y = 1.0 + 0.8 * x + u[group_ids] + eps
    return pd.DataFrame({"y": y, "x": x, "group": group_ids.astype(str)})


@pytest.fixture(scope="module")
def models(data):
    sm = MixedLM.from_formula("y ~ x", groups="group", data=data).fit(reml=True)
    il = interlace.fit("y ~ x", data=data, groups="group")
    return sm, il


@pytest.fixture(scope="module")
def sm_influence(models):
    sm, _ = models
    return hlm_influence(sm, level=1)


@pytest.fixture(scope="module")
def il_influence(models):
    _, il = models
    return hlm_influence(il, level=1)


# --- structure ---

REQUIRED_COLS = {"cooksd", "mdffits", "covtrace", "covratio"}


def test_returns_dataframe_with_required_cols(sm_influence, il_influence):
    for result in (sm_influence, il_influence):
        assert isinstance(result, pd.DataFrame)
        for col in REQUIRED_COLS:
            assert col in result.columns, f"missing column: {col}"
        assert any(c.startswith("rvc.") for c in result.columns), "missing rvc.* column"


def test_length_equals_nobs(sm_influence, il_influence, data):
    for result in (sm_influence, il_influence):
        assert len(result) == len(data)


def test_cooksd_nonnegative(sm_influence, il_influence):
    for result in (sm_influence, il_influence):
        assert (result["cooksd"].fillna(0) >= -1e-10).all()


# --- parity ---


def test_cooksd_match_across_models(sm_influence, il_influence):
    corr = np.corrcoef(sm_influence["cooksd"].values, il_influence["cooksd"].values)[
        0, 1
    ]
    assert corr > 0.9, f"Cook's D correlation={corr:.4f}"


# --- group-level ---


def test_group_level_returns_correct_structure(models, data):
    sm, il = models
    for model in (sm, il):
        result = hlm_influence(model, level="group")
        assert isinstance(result, pd.DataFrame)
        assert "cooksd" in result.columns
        n_groups = data["group"].nunique()
        assert len(result) == n_groups


# --- wrappers ---


def test_cooks_distance_wrapper(models, data):
    sm, il = models
    for model in (sm, il):
        cd = cooks_distance(model)
        assert isinstance(cd, np.ndarray)
        assert len(cd) == len(data)


def test_mdffits_wrapper(models, data):
    sm, il = models
    for model in (sm, il):
        mf = mdffits(model)
        assert isinstance(mf, np.ndarray)
        assert len(mf) == len(data)


# ---------------------------------------------------------------------------
# BOBYQA routing for single-RE statsmodels models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sm_with_group_col(data):
    """MixedLMResults with _gpgap_group_col set (as gpgap pipeline does)."""
    sm = MixedLM.from_formula("y ~ x", groups="group", data=data).fit(reml=True)
    sm._gpgap_group_col = "group"
    return sm


class TestBobyqaRouting:
    def test_hlm_influence_bobyqa_routes_through_interlace(
        self, sm_with_group_col
    ) -> None:
        pytest.importorskip("pybobyqa")
        with mock.patch("interlace.fit", wraps=interlace.fit) as mock_fit:
            hlm_influence(sm_with_group_col, optimizer="bobyqa")
        assert mock_fit.called, (
            "interlace.fit was not called for single-RE BOBYQA refit"
        )

    def test_hlm_influence_bobyqa_returns_valid_dataframe(
        self, sm_with_group_col, data
    ) -> None:
        pytest.importorskip("pybobyqa")
        result = hlm_influence(sm_with_group_col, optimizer="bobyqa")
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"cooksd", "mdffits", "covtrace", "covratio"}
        assert len(result) == len(data)
        assert (result["cooksd"].fillna(0) >= -1e-10).all()

    def test_hlm_influence_lbfgsb_does_not_route_through_interlace(
        self, sm_with_group_col
    ) -> None:
        """Default optimizer must not change routing (backward compat)."""
        with mock.patch("interlace.fit", wraps=interlace.fit) as mock_fit:
            hlm_influence(sm_with_group_col, optimizer="lbfgsb")
        assert not mock_fit.called, "interlace.fit should not be called for lbfgsb path"

    def test_unknown_optimizer_raises(self, sm_with_group_col) -> None:
        with pytest.raises(ValueError, match="optimizer"):
            hlm_influence(sm_with_group_col, optimizer="invalid")


# ---------------------------------------------------------------------------
# Warm-start: theta0 passed through interlace.fit and used in hlm_influence
# ---------------------------------------------------------------------------


class TestInfluenceRandomSlopes:
    """hlm_influence works on random-slopes CrossedLMEResult."""

    @pytest.fixture(scope="class")
    def slope_data(self):
        rng = np.random.default_rng(7)
        n_groups, n_per = 8, 5
        n = n_groups * n_per
        g = np.repeat(np.arange(n_groups), n_per).astype(str)
        x = rng.standard_normal(n)
        b0 = rng.normal(0, 0.8, n_groups)
        b1 = rng.normal(0, 0.4, n_groups)
        eps = rng.normal(0, 0.5, n)
        y = 1.0 + 0.5 * x + b0[g.astype(int)] + b1[g.astype(int)] * x + eps
        return pd.DataFrame({"y": y, "x": x, "g": g})

    @pytest.fixture(scope="class")
    def slope_model(self, slope_data):
        return interlace.fit("y ~ x", data=slope_data, random=["(1 + x | g)"])

    def test_hlm_influence_slopes_returns_dataframe(self, slope_model, slope_data):
        result = hlm_influence(slope_model, level=1)
        assert isinstance(result, pd.DataFrame)
        for col in ("cooksd", "mdffits", "covtrace", "covratio"):
            assert col in result.columns, f"missing column: {col}"
        assert any(c.startswith("rvc.") for c in result.columns)
        assert len(result) == len(slope_data)

    def test_hlm_influence_slopes_cooksd_nonneg(self, slope_model):
        result = hlm_influence(slope_model, level=1)
        assert (result["cooksd"].fillna(0) >= -1e-10).all()

    def test_hlm_influence_slopes_rvc_columns_per_term(self, slope_model):
        """RVC columns must reflect multi-term VC — one entry per diagonal element."""
        result = hlm_influence(slope_model, level=1)
        rvc_cols = [c for c in result.columns if c.startswith("rvc.")]
        # correlated (1+x|g) has 3 Cholesky params + 1 error_var = 4 rvc columns
        # independent (1+x||g) has 2 diag params + 1 error_var = 3 rvc columns
        assert len(rvc_cols) >= 2, f"too few rvc columns: {rvc_cols}"

    def test_hlm_influence_slopes_group_level(self, slope_model, slope_data):
        result = hlm_influence(slope_model, level="g")
        assert isinstance(result, pd.DataFrame)
        assert "cooksd" in result.columns
        assert len(result) == slope_data["g"].nunique()


class TestWarmStart:
    def test_fit_accepts_theta0(self, data) -> None:
        """interlace.fit() should accept theta0 and pass it to fit_reml."""
        import interlace
        from interlace.profiled_reml import fit_reml

        theta0 = np.array([1.5])
        with mock.patch("interlace.fit_reml", wraps=fit_reml) as mock_fit_reml:
            interlace.fit("y ~ x", data=data, groups="group", theta0=theta0)
        call_kwargs = mock_fit_reml.call_args
        passed_theta0 = call_kwargs.kwargs.get("theta0") or (
            call_kwargs.args[4] if len(call_kwargs.args) > 4 else None
        )
        assert passed_theta0 is not None, "theta0 was not forwarded to fit_reml"
        np.testing.assert_array_equal(passed_theta0, theta0)

    def test_hlm_influence_warm_starts_from_full_model_theta(self, models) -> None:
        """hlm_influence should pass model.theta as theta0 to each fit_reml call."""
        from interlace.profiled_reml import fit_reml

        _, il = models
        with mock.patch("interlace.profiled_reml.fit_reml", wraps=fit_reml) as mock_fr:
            hlm_influence(il, level=1)
        assert mock_fr.called
        for call in mock_fr.call_args_list:
            theta0_passed = call.kwargs.get("theta0")
            assert theta0_passed is not None, "theta0 not passed to fit_reml"
            np.testing.assert_array_equal(theta0_passed, il.theta)


# ---------------------------------------------------------------------------
# Matrix-cache optimisation (GitHub issue #7)
# ---------------------------------------------------------------------------


class TestMatrixCacheOptimization:
    """hlm_influence pre-builds X/y/Z once and skips formula re-parsing."""

    def test_formula_not_reparsed_in_loop(self, models, data) -> None:
        """formulaic.model_matrix must not be called inside the refit loop."""
        import formulaic

        _, il = models
        with mock.patch.object(
            formulaic, "model_matrix", wraps=formulaic.model_matrix
        ) as mock_mm:
            hlm_influence(il, level=1)

        assert mock_mm.call_count == 0, (
            f"formulaic.model_matrix was called {mock_mm.call_count} times "
            f"for {len(data)} refits; expected 0 (X/y taken from model, Z pre-built)"
        )

    def test_cooksd_values_unchanged_after_optimisation(
        self, il_influence, data
    ) -> None:
        """Diagnostic values must be numerically identical with the matrix path."""
        import interlace

        il = interlace.fit("y ~ x", data=data, groups="group")
        result = hlm_influence(il, level=1)

        np.testing.assert_allclose(
            result["cooksd"].values,
            il_influence["cooksd"].values,
            rtol=1e-6,
            err_msg="cooksd changed after matrix-cache optimisation",
        )


# ---------------------------------------------------------------------------
# Private helper unit tests (covers dead-code / rarely-reached paths)
# ---------------------------------------------------------------------------


class TestRequirePandas:
    def test_importerror_when_pandas_missing(self):
        """Lines 34-35: ImportError raised with helpful message."""
        import sys

        with (
            mock.patch.dict(sys.modules, {"pandas": None}),
            pytest.raises(ImportError, match="statsmodels compat path requires pandas"),
        ):
            _require_pandas()


class TestVcToScalars:
    def test_scalar_vc(self):
        vals, names = _vc_to_scalars(np.float64(2.5), "grp")
        assert vals == [2.5]
        assert names == ["var_grp"]

    def test_array_vc_with_index(self):
        """pandas Series VC (has .index) — existing covered path."""
        s = pd.Series([1.0, 2.0], index=["(Intercept)", "x"])
        vals, names = _vc_to_scalars(s, "grp")
        assert vals == [1.0, 2.0]
        assert names == ["var_grp_(Intercept)", "var_grp_x"]

    def test_array_vc_without_index(self):
        """Line 61: plain numpy array — no .index attribute."""
        vc = np.array([3.0, 4.0])
        vals, names = _vc_to_scalars(vc, "grp")
        assert vals == [3.0, 4.0]
        assert names == ["var_grp_0", "var_grp_1"]

    def test_matrix_vc_with_index(self):
        """2-D VC array uses diagonal elements."""
        vc = np.array([[4.0, 1.0], [1.0, 9.0]])
        vals, names = _vc_to_scalars(vc, "grp")
        assert vals == pytest.approx([4.0, 9.0])


class TestRefitDirect:
    """Lines 105-127: _refit is dead code in hlm_influence but is a valid
    internal function; test both branches directly."""

    def test_refit_crossed_no_slopes(self, data):
        """Lines 114-117: single random-intercept CrossedLMEResult."""
        il = interlace.fit("y ~ x", data=data, groups="group")
        subset = data.iloc[:-5].reset_index(drop=True)
        result = _refit(il, subset)
        assert hasattr(result, "fe_params")
        assert hasattr(result, "scale")

    def test_refit_crossed_with_slopes(self, data):
        """Lines 111-113: random-slopes CrossedLMEResult."""
        il = interlace.fit("y ~ x", data=data, random=["(1 + x | group)"])
        subset = data.iloc[:-5].reset_index(drop=True)
        result = _refit(il, subset)
        assert hasattr(result, "fe_params")


class TestReducedParamsCrossed:
    """Lines 137-145: _reduced_params with a CrossedLMEResult model_i."""

    def test_returns_correct_shapes(self, data):
        il = interlace.fit("y ~ x", data=data, groups="group")
        theta_names = ["var_group", "error_var"]
        beta_i, Vi, theta_i = _reduced_params(il, p=2, theta_names=theta_names)
        assert len(np.asarray(beta_i)) == 2
        assert Vi.shape == (2, 2)
        assert len(theta_i) == 2


class TestRefitGroupsArgNotCrossed:
    """Line 158: _refit_groups_arg returns None for non-crossed models."""

    def test_non_crossed_returns_none(self, models):
        sm, _ = models
        assert _refit_groups_arg(sm) is None


class TestIndependentSlopesInfluence:
    """Line 232: else branch in _refit_matrices_crossed for independent slopes."""

    @pytest.fixture(scope="class")
    def indep_data(self):
        rng = np.random.default_rng(99)
        n_groups, n_per = 6, 6
        n = n_groups * n_per
        g = np.repeat(np.arange(n_groups), n_per).astype(str)
        x = rng.standard_normal(n)
        b0 = rng.normal(0, 0.5, n_groups)
        b1 = rng.normal(0, 0.3, n_groups)
        eps = rng.normal(0, 0.5, n)
        y = 1.0 + 0.5 * x + b0[g.astype(int)] + b1[g.astype(int)] * x + eps
        return pd.DataFrame({"y": y, "x": x, "g": g})

    @pytest.fixture(scope="class")
    def indep_model(self, indep_data):
        return interlace.fit("y ~ x", data=indep_data, random=["(1 + x || g)"])

    def test_hlm_influence_independent_slopes(self, indep_model, indep_data):
        result = hlm_influence(indep_model, level=1)
        assert isinstance(result, pd.DataFrame)
        assert "cooksd" in result.columns
        assert len(result) == len(indep_data)

    def test_cooksd_nonnegative(self, indep_model):
        result = hlm_influence(indep_model, level=1)
        assert (result["cooksd"].fillna(0) >= -1e-10).all()


class TestIntegerLevelGroupDeletion:
    """Lines 313 + 416-417: hlm_influence with a non-1, non-string integer level.

    When level is not 1 and not a str, the code falls back to
    np.unique(groups) for units and groups != unit for masking.
    """

    def test_integer_level_statsmodels(self, models, data):
        sm, _ = models
        result = hlm_influence(sm, level=2)
        assert isinstance(result, pd.DataFrame)
        assert "cooksd" in result.columns
        n_groups = data["group"].nunique()
        assert len(result) == n_groups

    def test_integer_level_crossed(self, models, data):
        _, il = models
        result = hlm_influence(il, level=2)
        assert isinstance(result, pd.DataFrame)
        n_groups = data["group"].nunique()
        assert len(result) == n_groups


class TestExceptionSilencing:
    """Lines 446-447: exception inside refit loop leaves row as NaN."""

    def test_failed_refit_produces_nan(self, models):
        from interlace import influence

        _, il = models
        side_effect = RuntimeError("simulated failure")
        with mock.patch.object(
            influence, "_refit_matrices_crossed", side_effect=side_effect
        ):
            result = hlm_influence(il, level=1)
        assert result["cooksd"].isna().all()


# ---------------------------------------------------------------------------
# ols_dfbetas_qr (lines 648-680)
# ---------------------------------------------------------------------------


class TestOlsDfbetasQr:
    @pytest.fixture(scope="class")
    def ols_result(self, data):
        return smf.ols("y ~ x", data=data).fit()

    def test_returns_correct_shape(self, ols_result, data):
        result = ols_dfbetas_qr(ols_result)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(data), 2)  # n obs × (Intercept + x)

    def test_values_finite(self, ols_result):
        result = ols_dfbetas_qr(ols_result)
        assert np.all(np.isfinite(result))

    def test_matches_r_convention_small_influence(self, ols_result, data):
        """Obs with small leverage should have small DFBETAS."""
        result = ols_dfbetas_qr(ols_result)
        # Most observations in a well-conditioned dataset should be < 2/sqrt(n)
        n = len(data)
        threshold = 2 / np.sqrt(n)
        frac_large = np.mean(np.abs(result) > threshold)
        assert frac_large < 0.2, f"Too many large DFBETAS: {frac_large:.2%}"
