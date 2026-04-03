"""Tests for GLMM Laplace approximation (glmm_laplace.py).

Tests follow TDD: written before the implementation exists.

Coverage:
1. Gaussian family reduces to LMM (sanity check)
2. Binomial GLMM parity with lme4 on cbpp dataset
3. Poisson GLMM parity with lme4 on simulated data
4. PIRLS convergence and basic properties
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import interlace
from interlace.glmm_family import BinomialFamily, PoissonFamily
from interlace.glmm_laplace import GLMMResult, fit_glmm

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cbpp_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "glmm_cbpp_data.csv")


@pytest.fixture(scope="module")
def cbpp_ref() -> dict:
    return json.loads((FIXTURES / "glmm_cbpp_results.json").read_text())


@pytest.fixture(scope="module")
def cbpp_fit(cbpp_data) -> GLMMResult:
    """Fit binomial GLMM on cbpp: proportion ~ C(period) + (1|herd), weights=size."""
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
def poisson_ref() -> dict:
    return json.loads((FIXTURES / "glmm_poisson_results.json").read_text())


@pytest.fixture(scope="module")
def poisson_fit(poisson_data) -> GLMMResult:
    """Fit Poisson GLMM: y ~ x + (1|group)."""
    return fit_glmm(
        formula="y ~ x",
        data=poisson_data,
        family="poisson",
        groups="group",
    )


# ---------------------------------------------------------------------------
# 1. Gaussian sanity: GLMM with Gaussian family ≈ LMM
# ---------------------------------------------------------------------------


class TestGaussianSanity:
    """GLMM with Gaussian family should give similar results to fit()."""

    @pytest.fixture(scope="class")
    def dyestuff_data(self) -> pd.DataFrame:
        return pd.read_csv(FIXTURES / "lme4_dyestuff_data.csv")

    @pytest.fixture(scope="class")
    def lmm_fit(self, dyestuff_data):
        return interlace.fit("Yield ~ 1", data=dyestuff_data, groups="Batch")

    @pytest.fixture(scope="class")
    def glmm_gaussian_fit(self, dyestuff_data):
        return fit_glmm(
            formula="Yield ~ 1",
            data=dyestuff_data,
            family="gaussian",
            groups="Batch",
        )

    def test_fixed_effects_close(self, lmm_fit, glmm_gaussian_fit):
        """Gaussian GLMM fixed effects should match LMM closely."""
        lmm_beta = lmm_fit.fe_params.values
        glmm_beta = glmm_gaussian_fit.fe_params.values
        assert_allclose(glmm_beta, lmm_beta, rtol=0.05)

    def test_variance_components_close(self, lmm_fit, glmm_gaussian_fit):
        """Gaussian GLMM variance components should match LMM closely."""
        lmm_vc = lmm_fit.variance_components["Batch"]
        glmm_vc = glmm_gaussian_fit.variance_components["Batch"]
        assert_allclose(glmm_vc, lmm_vc, rtol=0.15)


# ---------------------------------------------------------------------------
# 2. Binomial GLMM: parity with lme4 on cbpp
# ---------------------------------------------------------------------------


# Map R coefficient names → formulaic names
_CBPP_NAME_MAP = {
    "(Intercept)": "Intercept",
    "period2": "C(period)[T.2]",
    "period3": "C(period)[T.3]",
    "period4": "C(period)[T.4]",
}


class TestBinomialCBPP:
    """Binomial GLMM on cbpp dataset should match lme4::glmer()."""

    def test_converged(self, cbpp_fit):
        assert cbpp_fit.converged

    def test_fixed_effects(self, cbpp_fit, cbpp_ref):
        """Fixed effects within 0.05 of lme4."""
        ref_fe = cbpp_ref["fixed_effects"]
        for r_name, ref_val in ref_fe.items():
            py_name = _CBPP_NAME_MAP[r_name]
            assert abs(cbpp_fit.fe_params[py_name] - ref_val) < 0.05, (
                f"{py_name}: interlace={cbpp_fit.fe_params[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_fixed_effects_se(self, cbpp_fit, cbpp_ref):
        """Fixed effects SEs within 10% of lme4."""
        ref_se = cbpp_ref["fixed_effects_se"]
        for r_name, ref_val in ref_se.items():
            py_name = _CBPP_NAME_MAP[r_name]
            assert abs(cbpp_fit.fe_bse[py_name] - ref_val) / ref_val < 0.10, (
                f"{py_name}: interlace={cbpp_fit.fe_bse[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_variance_component(self, cbpp_fit, cbpp_ref):
        """Herd variance within 15% of lme4."""
        ref_vc = cbpp_ref["variance_components"]["herd"]
        fit_vc = cbpp_fit.variance_components["herd"]
        assert abs(fit_vc - ref_vc) / ref_vc < 0.15, (
            f"herd VC: interlace={fit_vc:.4f}, lme4={ref_vc:.4f}"
        )

    def test_loglik(self, cbpp_fit, cbpp_ref):
        """Log-likelihood within 1.0 of lme4."""
        assert abs(cbpp_fit.llf - cbpp_ref["loglik"]) < 1.0, (
            f"loglik: interlace={cbpp_fit.llf:.2f}, lme4={cbpp_ref['loglik']:.2f}"
        )

    def test_random_effects_correlation(self, cbpp_fit, cbpp_ref):
        """BLUPs should correlate > 0.95 with lme4."""
        ref_re = np.array(cbpp_ref["random_effects_herd"])
        fit_re = cbpp_fit.random_effects["herd"]
        # Sort by index to align (R uses string "1".."15", Python may use int)
        fit_vals = fit_re.sort_index().values
        corr = np.corrcoef(fit_vals, ref_re)[0, 1]
        assert corr > 0.95, f"BLUP correlation = {corr:.4f}"

    def test_result_has_family(self, cbpp_fit):
        assert isinstance(cbpp_fit.family, BinomialFamily)

    def test_nobs(self, cbpp_fit, cbpp_ref):
        assert cbpp_fit.nobs == cbpp_ref["nobs"]


# ---------------------------------------------------------------------------
# 3. Poisson GLMM parity with lme4
# ---------------------------------------------------------------------------


class TestPoissonGLMM:
    """Poisson GLMM should match lme4::glmer()."""

    def test_converged(self, poisson_fit):
        assert poisson_fit.converged

    def test_fixed_effects(self, poisson_fit, poisson_ref):
        """Fixed effects within 0.05 of lme4."""
        ref_fe = poisson_ref["fixed_effects"]
        name_map = {"(Intercept)": "Intercept", "x": "x"}
        for r_name, ref_val in ref_fe.items():
            py_name = name_map[r_name]
            assert abs(poisson_fit.fe_params[py_name] - ref_val) < 0.05, (
                f"{py_name}: interlace={poisson_fit.fe_params[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_variance_component(self, poisson_fit, poisson_ref):
        """Group variance within 20% of lme4."""
        ref_vc = poisson_ref["variance_components"]["group"]
        fit_vc = poisson_fit.variance_components["group"]
        assert abs(fit_vc - ref_vc) / ref_vc < 0.20, (
            f"group VC: interlace={fit_vc:.4f}, lme4={ref_vc:.4f}"
        )

    def test_loglik(self, poisson_fit, poisson_ref):
        """Log-likelihood within 2.0 of lme4."""
        assert abs(poisson_fit.llf - poisson_ref["loglik"]) < 2.0, (
            f"loglik: interlace={poisson_fit.llf:.2f}, lme4={poisson_ref['loglik']:.2f}"
        )

    def test_result_has_family(self, poisson_fit):
        assert isinstance(poisson_fit.family, PoissonFamily)


# ---------------------------------------------------------------------------
# 4. GLMMResult basic properties
# ---------------------------------------------------------------------------


class TestGLMMResultProperties:
    """GLMMResult should expose the expected attributes."""

    def test_has_fe_params(self, cbpp_fit):
        assert hasattr(cbpp_fit, "fe_params")
        assert len(cbpp_fit.fe_params) == 4  # intercept + 3 period dummies

    def test_has_fe_bse(self, cbpp_fit):
        assert hasattr(cbpp_fit, "fe_bse")
        assert len(cbpp_fit.fe_bse) == 4

    def test_has_random_effects(self, cbpp_fit):
        assert "herd" in cbpp_fit.random_effects

    def test_has_theta(self, cbpp_fit):
        assert hasattr(cbpp_fit, "theta")
        assert len(cbpp_fit.theta) >= 1

    def test_has_aic_bic(self, cbpp_fit):
        assert hasattr(cbpp_fit, "aic")
        assert hasattr(cbpp_fit, "bic")
        assert np.isfinite(cbpp_fit.aic)
        assert np.isfinite(cbpp_fit.bic)

    def test_has_converged(self, cbpp_fit):
        assert isinstance(cbpp_fit.converged, bool)
