"""Regression tests: GLMM Phase-2 Nelder-Mead aligned with lme4 conventions.

interlace runs a two-phase optimisation for GLMM: Phase 1 (L-BFGS-B over
theta, beta profiled) followed by Phase 2 (Nelder-Mead over (theta, beta)
jointly).  This mirrors lme4's two-stage glmer optimisation
(R/lmer.R:154-198).  These tests guard against drift away from lme4's
stage-2 conventions.

Reference: lme4 R/optimizer.R:27-32 (Nelder_Mead defaults FtolAbs=1e-5,
FtolRel=1e-15, XtolRel=1e-7, maxfun=10000).  Empirical lme4 on CBPP:
285 stage-2 evals (median over 50 reps on this host).
"""

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace import glmm_laplace

FIXTURES_DIR = "tests/fixtures"


@pytest.fixture
def cbpp_data() -> pd.DataFrame:
    df = pd.read_csv(f"{FIXTURES_DIR}/glmm_cbpp_data.csv")
    df["period"] = df["period"].astype(str)
    df["herd"] = df["herd"].astype(str)
    return df


def _count_phase2_evals(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counter = {"n": 0}
    orig = glmm_laplace._laplace_objective_profiled

    def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(glmm_laplace, "_laplace_objective_profiled", counted)
    return counter


class TestPhase2EvalCountAlignedWithLme4:
    """Phase-2 eval count on CBPP must stay close to lme4's 285."""

    def test_cbpp_phase2_evals_within_lme4_envelope(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # lme4 stage-2 evals on CBPP: 285 (empirical, single fit).
        # 1.5x envelope = 430.  Tighter target 1.23x = 350: with lme4's
        # FtolAbs=1e-5 we expect to land around 200-300.
        counter = _count_phase2_evals(monkeypatch)
        interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
        )
        assert counter["n"] <= 350, (
            f"Phase-2 eval count {counter['n']} exceeds 1.23x of lme4's 285. "
            "If this regressed, check glmm_laplace.py Phase-2 options "
            "(fatol, xatol, adaptive) against lme4 R/optimizer.R defaults."
        )

    def test_cbpp_phase2_evals_above_zero(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity: Phase 2 is actually running on CBPP (nAGQ=1, binomial)."""
        counter = _count_phase2_evals(monkeypatch)
        interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
        )
        assert counter["n"] > 0, "Phase 2 should run on default CBPP fit"


class TestNumericalAgreementPreserved:
    """After loosening Phase-2 tolerances, CBPP estimates must still match lme4."""

    def test_cbpp_fixed_effects_match_lme4(self, cbpp_data: pd.DataFrame) -> None:
        # lme4 reference values for CBPP (from manuscript validation suite,
        # commit 19f7a8 in interlace-manuscript).  Wider tolerance than the
        # validation suite (1e-3 abs) since we're only guarding regression
        # from Phase-2 tolerance changes here.
        result = interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
        )
        # lme4 estimates: Intercept=-1.398, period[T.2]=-0.992,
        # period[T.3]=-1.128, period[T.4]=-1.580
        np.testing.assert_allclose(
            result.fe_params["Intercept"], -1.398, atol=5e-2
        )
        np.testing.assert_allclose(
            result.fe_params["period[T.2]"], -0.992, atol=5e-2
        )
        np.testing.assert_allclose(
            result.fe_params["period[T.3]"], -1.128, atol=5e-2
        )
        np.testing.assert_allclose(
            result.fe_params["period[T.4]"], -1.580, atol=5e-2
        )

    def test_cbpp_variance_component_matches_lme4(
        self, cbpp_data: pd.DataFrame
    ) -> None:
        # lme4 herd variance ~= 0.4123 (sd 0.6422, theta 0.6422)
        result = interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
        )
        herd_var = result.variance_components["herd"]
        np.testing.assert_allclose(herd_var, 0.4123, rtol=0.10)


class TestNAGQBypassesPhase2:
    """Phase 2 should NOT run when nAGQ >= 2 (AGQ replaces the joint search,
    matching lme4's behaviour where stage 2 changes math at nAGQ > 1)."""

    def test_nagq_2_skips_phase2(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        counter = _count_phase2_evals(monkeypatch)
        interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
            nAGQ=5,
        )
        assert counter["n"] == 0, (
            f"Phase 2 ran ({counter['n']} evals) under nAGQ=5; should be "
            "bypassed in favour of AGQ quadrature"
        )
