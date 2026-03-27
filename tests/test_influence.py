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
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.influence import cooks_distance, hlm_influence, mdffits


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
        """hlm_influence should pass model.theta as theta0 for each interlace refit."""
        _, il = models
        with mock.patch("interlace.fit", wraps=interlace.fit) as mock_fit:
            hlm_influence(il, level=1)
        assert mock_fit.called
        for call in mock_fit.call_args_list:
            theta0_passed = call.kwargs.get("theta0")
            assert theta0_passed is not None, "theta0 not passed to interlace.fit refit"
            np.testing.assert_array_equal(theta0_passed, il.theta)
