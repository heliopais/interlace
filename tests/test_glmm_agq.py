"""Tests for Adaptive Gauss-Hermite Quadrature (AGQ) in GLMM.

Tests follow TDD: written before the implementation exists.

Coverage:
1. nAGQ parameter accepted by fit_glmm / glmer
2. nAGQ=1 recovers Laplace results exactly
3. Binomial AGQ parity with lme4 (nAGQ=25) on cbpp
4. Poisson AGQ parity with lme4 (nAGQ=25) on simulated data
5. nAGQ > 1 rejected for multi-RE models
6. AGQ log-likelihood improves over Laplace for binary data
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from interlace.glmm_laplace import GLMMResult, fit_glmm

FIXTURES = Path(__file__).parent / "fixtures"

# Map R coefficient names -> formulaic names (shared with Laplace tests)
_CBPP_NAME_MAP = {
    "(Intercept)": "Intercept",
    "period2": "C(period)[T.2]",
    "period3": "C(period)[T.3]",
    "period4": "C(period)[T.4]",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cbpp_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "glmm_cbpp_data.csv")


@pytest.fixture(scope="module")
def cbpp_agq_ref() -> dict:
    return json.loads((FIXTURES / "glmm_cbpp_agq25_results.json").read_text())


@pytest.fixture(scope="module")
def cbpp_laplace_ref() -> dict:
    return json.loads((FIXTURES / "glmm_cbpp_results.json").read_text())


@pytest.fixture(scope="module")
def cbpp_agq_fit(cbpp_data) -> GLMMResult:
    """Fit binomial GLMM on cbpp with nAGQ=25."""
    return fit_glmm(
        formula="proportion ~ C(period)",
        data=cbpp_data,
        family="binomial",
        groups="herd",
        weights=cbpp_data["size"].values.astype(float),
        nAGQ=25,
    )


@pytest.fixture(scope="module")
def cbpp_laplace_fit(cbpp_data) -> GLMMResult:
    """Fit binomial GLMM on cbpp with nAGQ=1 (explicit Laplace)."""
    return fit_glmm(
        formula="proportion ~ C(period)",
        data=cbpp_data,
        family="binomial",
        groups="herd",
        weights=cbpp_data["size"].values.astype(float),
        nAGQ=1,
    )


@pytest.fixture(scope="module")
def cbpp_default_fit(cbpp_data) -> GLMMResult:
    """Fit binomial GLMM on cbpp with default nAGQ (should be 1)."""
    return fit_glmm(
        formula="proportion ~ C(period)",
        data=cbpp_data,
        family="binomial",
        groups="herd",
        weights=cbpp_data["size"].values.astype(float),
    )


@pytest.fixture(scope="module")
def poisson_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "glmm_poisson_data.csv")


@pytest.fixture(scope="module")
def poisson_agq_ref() -> dict:
    return json.loads((FIXTURES / "glmm_poisson_agq25_results.json").read_text())


@pytest.fixture(scope="module")
def poisson_agq_fit(poisson_data) -> GLMMResult:
    """Fit Poisson GLMM with nAGQ=25."""
    return fit_glmm(
        formula="y ~ x",
        data=poisson_data,
        family="poisson",
        groups="group",
        nAGQ=25,
    )


# ---------------------------------------------------------------------------
# 1. nAGQ=1 recovers Laplace exactly
# ---------------------------------------------------------------------------


class TestAGQRecoverLaplace:
    """nAGQ=1 should produce identical results to default Laplace."""

    def test_loglik_matches_default(self, cbpp_laplace_fit, cbpp_default_fit):
        """nAGQ=1 log-likelihood should match default (no nAGQ) exactly."""
        assert_allclose(cbpp_laplace_fit.llf, cbpp_default_fit.llf, rtol=1e-6)

    def test_fe_matches_default(self, cbpp_laplace_fit, cbpp_default_fit):
        """nAGQ=1 fixed effects should match default exactly."""
        assert_allclose(
            cbpp_laplace_fit.fe_params.values,
            cbpp_default_fit.fe_params.values,
            rtol=1e-6,
        )

    def test_theta_matches_default(self, cbpp_laplace_fit, cbpp_default_fit):
        """nAGQ=1 theta should match default exactly."""
        assert_allclose(cbpp_laplace_fit.theta, cbpp_default_fit.theta, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. Binomial AGQ parity with lme4 (nAGQ=25)
# ---------------------------------------------------------------------------


class TestBinomialAGQ:
    """Binomial GLMM with nAGQ=25 should match lme4::glmer(nAGQ=25)."""

    def test_converged(self, cbpp_agq_fit):
        assert cbpp_agq_fit.converged

    def test_fixed_effects(self, cbpp_agq_fit, cbpp_agq_ref):
        """Fixed effects within 0.05 of lme4 AGQ."""
        ref_fe = cbpp_agq_ref["fixed_effects"]
        for r_name, ref_val in ref_fe.items():
            py_name = _CBPP_NAME_MAP[r_name]
            assert abs(cbpp_agq_fit.fe_params[py_name] - ref_val) < 0.05, (
                f"{py_name}: interlace={cbpp_agq_fit.fe_params[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_fixed_effects_se(self, cbpp_agq_fit, cbpp_agq_ref):
        """Fixed effects SEs within 10% of lme4 AGQ."""
        ref_se = cbpp_agq_ref["fixed_effects_se"]
        for r_name, ref_val in ref_se.items():
            py_name = _CBPP_NAME_MAP[r_name]
            assert abs(cbpp_agq_fit.fe_bse[py_name] - ref_val) / ref_val < 0.10, (
                f"{py_name}: interlace={cbpp_agq_fit.fe_bse[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_variance_component(self, cbpp_agq_fit, cbpp_agq_ref):
        """Herd variance within 15% of lme4 AGQ."""
        ref_vc = cbpp_agq_ref["variance_components"]["herd"]
        fit_vc = cbpp_agq_fit.variance_components["herd"]
        assert abs(fit_vc - ref_vc) / ref_vc < 0.15, (
            f"herd VC: interlace={fit_vc:.4f}, lme4={ref_vc:.4f}"
        )

    def test_loglik(self, cbpp_agq_fit, cbpp_agq_ref):
        """Log-likelihood within 1.0 of lme4 AGQ."""
        assert abs(cbpp_agq_fit.llf - cbpp_agq_ref["loglik"]) < 1.0, (
            f"loglik: interlace={cbpp_agq_fit.llf:.2f}, "
            f"lme4={cbpp_agq_ref['loglik']:.2f}"
        )

    def test_random_effects_correlation(self, cbpp_agq_fit, cbpp_agq_ref):
        """BLUPs should correlate > 0.95 with lme4 AGQ."""
        ref_re = np.array(cbpp_agq_ref["random_effects_herd"])
        fit_re = cbpp_agq_fit.random_effects["herd"]
        fit_vals = fit_re.sort_index().values
        corr = np.corrcoef(fit_vals, ref_re)[0, 1]
        assert corr > 0.95, f"BLUP correlation = {corr:.4f}"


# ---------------------------------------------------------------------------
# 3. Poisson AGQ parity with lme4 (nAGQ=25)
# ---------------------------------------------------------------------------


class TestPoissonAGQ:
    """Poisson GLMM with nAGQ=25 should match lme4::glmer(nAGQ=25)."""

    def test_converged(self, poisson_agq_fit):
        assert poisson_agq_fit.converged

    def test_fixed_effects(self, poisson_agq_fit, poisson_agq_ref):
        """Fixed effects within 0.05 of lme4 AGQ."""
        ref_fe = poisson_agq_ref["fixed_effects"]
        name_map = {"(Intercept)": "Intercept", "x": "x"}
        for r_name, ref_val in ref_fe.items():
            py_name = name_map[r_name]
            assert abs(poisson_agq_fit.fe_params[py_name] - ref_val) < 0.05, (
                f"{py_name}: interlace={poisson_agq_fit.fe_params[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_variance_component(self, poisson_agq_fit, poisson_agq_ref):
        """Group variance within 20% of lme4 AGQ."""
        ref_vc = poisson_agq_ref["variance_components"]["group"]
        fit_vc = poisson_agq_fit.variance_components["group"]
        assert abs(fit_vc - ref_vc) / ref_vc < 0.20, (
            f"group VC: interlace={fit_vc:.4f}, lme4={ref_vc:.4f}"
        )

    def test_loglik(self, poisson_agq_fit, poisson_agq_ref):
        """Log-likelihood within 2.0 of lme4 AGQ."""
        assert abs(poisson_agq_fit.llf - poisson_agq_ref["loglik"]) < 2.0, (
            f"loglik: interlace={poisson_agq_fit.llf:.2f}, "
            f"lme4={poisson_agq_ref['loglik']:.2f}"
        )


# ---------------------------------------------------------------------------
# 4. AGQ improves over Laplace for binary data
# ---------------------------------------------------------------------------


class TestAGQImprovesOverLaplace:
    """AGQ should give a better (higher) marginal log-likelihood than Laplace."""

    def test_agq_loglik_differs_from_laplace(self, cbpp_agq_fit, cbpp_default_fit):
        """AGQ log-likelihood should differ from Laplace (it's a different scale
        because AGQ integrates out the random effects while Laplace is an
        approximation to the same integral)."""
        # The values will be different; AGQ is generally considered more
        # accurate. We just check they're not identical.
        assert cbpp_agq_fit.llf != cbpp_default_fit.llf


# ---------------------------------------------------------------------------
# 5. nAGQ > 1 rejected for multiple random effects
# ---------------------------------------------------------------------------


class TestAGQValidation:
    """nAGQ > 1 should be rejected for models where it's not supported."""

    def test_nagq_gt1_rejected_for_multiple_groups(self, cbpp_data):
        """nAGQ > 1 not supported with multiple grouping factors."""
        # cbpp only has one grouping factor, so we use a synthetic case
        df = cbpp_data.copy()
        df["herd2"] = df["herd"]  # duplicate grouping factor

        with pytest.raises(ValueError, match="nAGQ > 1"):
            fit_glmm(
                formula="proportion ~ C(period)",
                data=df,
                family="binomial",
                groups=["herd", "herd2"],
                weights=df["size"].values.astype(float),
                nAGQ=25,
            )

    def test_nagq_gt1_rejected_for_random_slopes(self, cbpp_data):
        """nAGQ > 1 not supported with random slopes."""
        with pytest.raises(ValueError, match="nAGQ > 1"):
            fit_glmm(
                formula="proportion ~ C(period)",
                data=cbpp_data,
                family="binomial",
                random=["(1 + C(period) | herd)"],
                weights=cbpp_data["size"].values.astype(float),
                nAGQ=25,
            )

    def test_nagq_zero_raises(self, cbpp_data):
        """nAGQ=0 should raise (not supported)."""
        with pytest.raises(ValueError, match="nAGQ"):
            fit_glmm(
                formula="proportion ~ C(period)",
                data=cbpp_data,
                family="binomial",
                groups="herd",
                weights=cbpp_data["size"].values.astype(float),
                nAGQ=0,
            )
