"""Tests: GLMM hot path uses CHOLMOD when sksparse is available.

Before interlace-7sc3, all sparse solves in glmm_laplace.py went through
``spla.spsolve`` directly, bypassing the ``_try_cholmod`` machinery used
by profiled_reml.py for LMM.  This caused the CHOLMOD optional dep to
have 0% impact on GLMM/CBPP fits.
"""

from __future__ import annotations

import sys
import unittest.mock

import numpy as np
import pandas as pd
import pytest
import scipy.sparse.linalg as spla

import interlace
from interlace import glmm_laplace, profiled_reml

FIXTURES_DIR = "tests/fixtures"


@pytest.fixture
def cbpp_data() -> pd.DataFrame:
    df = pd.read_csv(f"{FIXTURES_DIR}/glmm_cbpp_data.csv")
    df["period"] = df["period"].astype(str)
    df["herd"] = df["herd"].astype(str)
    return df


def _has_sksparse() -> bool:
    return profiled_reml._try_cholmod() is not None


SKSPARSE_REQUIRED = pytest.mark.skipif(
    not _has_sksparse(),
    reason="sksparse not installed; CHOLMOD path tests need it",
)


def _fit_cbpp(df: pd.DataFrame):  # type: ignore[no-untyped-def]
    return interlace.glmer(
        "proportion ~ period",
        data=df,
        family="binomial",
        groups="herd",
        weights=np.array(df["size"], dtype=float),
    )


class TestSpsolveCountDropsWithCHOLMOD:
    """When sksparse is available, the GLMM hot path should not fall
    through to ``spla.spsolve`` for the inner-loop solves."""

    @SKSPARSE_REQUIRED
    def test_cbpp_spsolve_count_dramatically_reduced(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pre-7sc3: ~813 spsolve calls. Target: <= 50 (essentially only
        # fallbacks for edge cases / non-A solves elsewhere).
        counter = {"n": 0}
        orig = spla.spsolve

        def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
            counter["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(spla, "spsolve", counted)
        # Also patch the import in glmm_laplace's module namespace
        monkeypatch.setattr(glmm_laplace.spla, "spsolve", counted)

        _fit_cbpp(cbpp_data)
        assert counter["n"] <= 50, (
            f"Expected <=50 spla.spsolve calls with CHOLMOD enabled, got "
            f"{counter['n']}. Pre-7sc3 baseline was 813. If this regressed, "
            "the CHOLMOD factor probably isn't being passed through to "
            "_pirls / _laplace_objective_profiled."
        )


class TestNumericalAgreementBothPaths:
    """Fits with and without CHOLMOD must give the same answer."""

    @SKSPARSE_REQUIRED
    def test_cbpp_estimates_identical_with_and_without_cholmod(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Fit with CHOLMOD on (default when sksparse available)
        result_cholmod = _fit_cbpp(cbpp_data)

        # Fit with CHOLMOD forced off (mock _try_cholmod to return None)
        monkeypatch.setattr(profiled_reml, "_try_cholmod", lambda: None)
        # Also blow away any cached imports just in case
        monkeypatch.setitem(sys.modules, "sksparse", unittest.mock.MagicMock())
        monkeypatch.setitem(sys.modules, "sksparse.cholmod", unittest.mock.MagicMock())
        # Re-fetch _try_cholmod from glmm_laplace's module to ensure the
        # patched one is seen
        result_superlu = _fit_cbpp(cbpp_data)

        # FE params: tight tolerance — numerical refactorisation in CHOLMOD
        # vs SuperLU LU should agree to near machine precision on a
        # well-conditioned 5x5 problem.
        np.testing.assert_allclose(
            result_cholmod.fe_params.values,
            result_superlu.fe_params.values,
            atol=1e-6,
            rtol=1e-4,
        )
        # Variance components: looser since they go through nonlinear opt
        for key in result_cholmod.variance_components:
            np.testing.assert_allclose(
                result_cholmod.variance_components[key],
                result_superlu.variance_components[key],
                rtol=1e-3,
            )
        np.testing.assert_allclose(result_cholmod.llf, result_superlu.llf, atol=1e-4)


class TestSuperLUFallbackStillWorks:
    """When sksparse is unavailable (mocked or absent), GLMM still fits."""

    def test_cbpp_fits_with_cholmod_disabled(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(profiled_reml, "_try_cholmod", lambda: None)
        result = _fit_cbpp(cbpp_data)
        # Just check the fit succeeded and produced sensible numbers
        assert result.converged
        assert "Intercept" in result.fe_params
        np.testing.assert_allclose(result.fe_params["Intercept"], -1.398, atol=5e-2)
