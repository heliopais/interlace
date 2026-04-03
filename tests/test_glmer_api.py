"""Tests for the public interlace.glmer() API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace.glmm_family import BinomialFamily, PoissonFamily
from interlace.glmm_laplace import GLMMResult

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
def poisson_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "glmm_poisson_data.csv")


# ---------------------------------------------------------------------------
# API availability
# ---------------------------------------------------------------------------


class TestAPIAvailability:
    """glmer should be importable from the top-level package."""

    def test_glmer_in_all(self):
        assert "glmer" in interlace.__all__

    def test_glmer_callable(self):
        assert callable(interlace.glmer)

    def test_glmm_result_importable(self):
        from interlace import GLMMResult as _GLMMResult  # noqa: F401

    def test_glmm_family_importable(self):
        from interlace import GLMMFamily as _GLMMFamily  # noqa: F401


# ---------------------------------------------------------------------------
# Binomial via glmer()
# ---------------------------------------------------------------------------


class TestGlmerBinomial:
    """glmer() with family='binomial' on cbpp."""

    @pytest.fixture(scope="class")
    def result(self, cbpp_data):
        return interlace.glmer(
            formula="proportion ~ C(period)",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=cbpp_data["size"].values.astype(float),
        )

    def test_returns_glmm_result(self, result):
        assert isinstance(result, GLMMResult)

    def test_converged(self, result):
        assert result.converged

    def test_fe_count(self, result):
        assert len(result.fe_params) == 4

    def test_fe_close_to_lme4(self, result, cbpp_ref):
        ref_intercept = cbpp_ref["fixed_effects"]["(Intercept)"]
        assert abs(result.fe_params["Intercept"] - ref_intercept) < 0.05

    def test_family_attribute(self, result):
        assert isinstance(result.family, BinomialFamily)

    def test_has_random_effects(self, result):
        assert "herd" in result.random_effects
        assert len(result.random_effects["herd"]) == 15

    def test_has_info_criteria(self, result):
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert np.isfinite(result.llf)


# ---------------------------------------------------------------------------
# Poisson via glmer()
# ---------------------------------------------------------------------------


class TestGlmerPoisson:
    """glmer() with family='poisson'."""

    @pytest.fixture(scope="class")
    def result(self, poisson_data):
        return interlace.glmer(
            formula="y ~ x",
            data=poisson_data,
            family="poisson",
            groups="group",
        )

    def test_returns_glmm_result(self, result):
        assert isinstance(result, GLMMResult)

    def test_converged(self, result):
        assert result.converged

    def test_family_attribute(self, result):
        assert isinstance(result.family, PoissonFamily)


# ---------------------------------------------------------------------------
# Family instance passthrough
# ---------------------------------------------------------------------------


class TestFamilyPassthrough:
    """glmer() should accept a GLMMFamily instance, not just a string."""

    def test_family_instance(self, cbpp_data):
        fam = BinomialFamily()
        result = interlace.glmer(
            formula="proportion ~ C(period)",
            data=cbpp_data,
            family=fam,
            groups="herd",
            weights=cbpp_data["size"].values.astype(float),
        )
        assert result.family is fam


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestGlmerErrors:
    """glmer() should raise clear errors for invalid inputs."""

    def test_no_groups_raises(self, cbpp_data):
        with pytest.raises(ValueError, match="groups.*random"):
            interlace.glmer(
                formula="proportion ~ C(period)",
                data=cbpp_data,
                family="binomial",
            )

    def test_unknown_family_raises(self, cbpp_data):
        with pytest.raises(ValueError, match="Unknown family"):
            interlace.glmer(
                formula="proportion ~ C(period)",
                data=cbpp_data,
                family="gamma",
                groups="herd",
            )
