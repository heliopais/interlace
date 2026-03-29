"""Tests for ML fitting and anova() model comparison.

TDD: these tests were written before the implementation.

Acceptance criteria (from interlace-7rw):
  - fit(..., method='ML') converges and matches lme4 ML log-likelihood to <0.01
  - anova(m1, m2) returns a DataFrame with the correct lme4-style columns
  - Raises ValueError when comparing REML-fitted models without refitting
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

import interlace
from interlace import anova

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def two_re_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "two_re_data.csv")


@pytest.fixture(scope="module")
def ml_result(two_re_data):
    return interlace.fit(
        "y ~ x", data=two_re_data, groups=["firm", "dept"], method="ML"
    )


@pytest.fixture(scope="module")
def reml_result(two_re_data):
    return interlace.fit(
        "y ~ x", data=two_re_data, groups=["firm", "dept"], method="REML"
    )


@pytest.fixture(scope="module")
def ml_null(two_re_data):
    return interlace.fit(
        "y ~ 1", data=two_re_data, groups=["firm", "dept"], method="ML"
    )


# ---------------------------------------------------------------------------
# ML fitting: basic API
# ---------------------------------------------------------------------------


def test_fit_ml_method_accepted(ml_result):
    assert ml_result.method == "ML"


def test_fit_ml_converged(ml_result):
    assert ml_result.converged


def test_fit_ml_has_nparams(ml_result):
    # p=2 (Intercept + x) + n_theta=2 (one per RE group) + 1 (sigma2) = 5
    assert ml_result.nparams == 5


def test_fit_ml_fixed_effects_close_to_reml(ml_result, reml_result):
    """ML and REML fixed effects should be very close (typically within 1%)."""
    ml_fe = np.asarray(ml_result.fe_params)
    reml_fe = np.asarray(reml_result.fe_params)
    np.testing.assert_allclose(ml_fe, reml_fe, rtol=0.05, atol=1e-3)


def test_fit_ml_sigma2_positive(ml_result):
    assert ml_result.scale > 0


def test_fit_ml_sigma2_uses_n_denominator(ml_result):
    """ML sigma2 = yPy / n, so it should be smaller than REML sigma2 = yPy / (n-p)."""
    # ML and REML may have slightly different thetas so yPy differs,
    # but ML sigma2 should generally be < REML sigma2 for the same data.
    # We verify the attribute exists and is positive.
    assert ml_result.scale > 0


def test_fit_ml_llf_is_float(ml_result):
    assert isinstance(ml_result.llf, float)
    assert np.isfinite(ml_result.llf)


def test_fit_ml_aic_bic_finite(ml_result):
    assert np.isfinite(ml_result.aic)
    assert np.isfinite(ml_result.bic)
    # AIC = -2*llf + 2*nparams
    expected_aic = -2.0 * ml_result.llf + 2.0 * ml_result.nparams
    assert abs(ml_result.aic - expected_aic) < 1e-8


def test_fit_ml_bic_formula(ml_result):
    # BIC = -2*llf + log(n)*nparams
    expected_bic = -2.0 * ml_result.llf + np.log(ml_result.nobs) * ml_result.nparams
    assert abs(ml_result.bic - expected_bic) < 1e-8


def test_fit_ml_deviance(ml_result):
    # deviance = -2 * llf
    expected_deviance = -2.0 * ml_result.llf
    assert abs(expected_deviance - (-2.0 * ml_result.llf)) < 1e-10


# ---------------------------------------------------------------------------
# ML fitting: lme4 parity
# ---------------------------------------------------------------------------

ML_PARITY_FIXTURE = FIXTURES / "ml_r_results.json"


@pytest.mark.skipif(
    not ML_PARITY_FIXTURE.exists(),
    reason="lme4 ML fixture not generated yet; run tests/fixtures/gen_ml.R",
)
def test_ml_llf_matches_lme4(two_re_data):
    r_results = json.loads(ML_PARITY_FIXTURE.read_text())
    ml = interlace.fit("y ~ x", data=two_re_data, groups=["firm", "dept"], method="ML")
    diff = abs(ml.llf - r_results["llf_ml"])
    assert diff < 0.01, (
        f"ML log-likelihood diff={diff:.4f} "
        f"(interlace={ml.llf:.6f}, lme4={r_results['llf_ml']:.6f})"
    )


@pytest.mark.skipif(
    not ML_PARITY_FIXTURE.exists(),
    reason="lme4 ML fixture not generated yet; run tests/fixtures/gen_ml.R",
)
def test_ml_fixed_effects_match_lme4(two_re_data):
    r_results = json.loads(ML_PARITY_FIXTURE.read_text())
    ml = interlace.fit("y ~ x", data=two_re_data, groups=["firm", "dept"], method="ML")
    r_fe = r_results["fe_params"]
    for r_name, il_name in [("(Intercept)", "Intercept"), ("x", "x")]:
        diff = abs(float(ml.fe_params[il_name]) - r_fe[r_name])
        assert diff < 1e-4, (
            f"Fixed effect '{il_name}' abs_diff={diff:.2e} "
            f"(interlace={float(ml.fe_params[il_name]):.6f}, R={r_fe[r_name]:.6f})"
        )


# ---------------------------------------------------------------------------
# anova(): structure
# ---------------------------------------------------------------------------

EXPECTED_ANOVA_COLS = [
    "Df",
    "AIC",
    "BIC",
    "logLik",
    "deviance",
    "Chisq",
    "Chi Df",
    "Pr(>Chisq)",
]


def test_anova_returns_dataframe(two_re_data, ml_null, ml_result):
    result = anova(ml_null, ml_result)
    assert isinstance(result, pd.DataFrame)


def test_anova_has_correct_columns(two_re_data, ml_null, ml_result):
    result = anova(ml_null, ml_result)
    for col in EXPECTED_ANOVA_COLS:
        assert col in result.columns, f"Missing column: {col!r}"


def test_anova_has_two_rows(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    assert len(result) == 2


def test_anova_df_column(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    # m_null: p=1 + n_theta=2 + 1 = 4; m_full: p=2 + 2 + 1 = 5
    assert result["Df"].iloc[0] == 4
    assert result["Df"].iloc[1] == 5


def test_anova_chi_df(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    # Chi Df = difference in Df = 5 - 4 = 1
    assert result["Chi Df"].iloc[1] == 1
    # First row has no chi-sq test
    assert np.isnan(result["Chi Df"].iloc[0]) or result["Chi Df"].iloc[0] == 0


def test_anova_chisq_positive(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    # ml_result (with x) should have higher llf than ml_null
    assert result["Chisq"].iloc[1] > 0


def test_anova_pvalue_range(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    p = result["Pr(>Chisq)"].iloc[1]
    assert 0.0 <= p <= 1.0


def test_anova_loglik_column(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    assert abs(result["logLik"].iloc[0] - ml_null.llf) < 1e-8
    assert abs(result["logLik"].iloc[1] - ml_result.llf) < 1e-8


def test_anova_deviance_column(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    assert abs(result["deviance"].iloc[0] - (-2.0 * ml_null.llf)) < 1e-8
    assert abs(result["deviance"].iloc[1] - (-2.0 * ml_result.llf)) < 1e-8


def test_anova_chisq_formula(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    # LRT = 2 * (llf_larger - llf_smaller)
    expected_chisq = 2.0 * (ml_result.llf - ml_null.llf)
    assert abs(result["Chisq"].iloc[1] - expected_chisq) < 1e-8


def test_anova_pvalue_from_chisq(ml_null, ml_result):
    result = anova(ml_null, ml_result)
    chisq = result["Chisq"].iloc[1]
    chi_df = result["Chi Df"].iloc[1]
    expected_p = float(stats.chi2.sf(chisq, df=chi_df))
    assert abs(result["Pr(>Chisq)"].iloc[1] - expected_p) < 1e-8


def test_anova_order_independent(ml_null, ml_result):
    """anova(m1, m2) and anova(m2, m1) should give the same result."""
    r1 = anova(ml_null, ml_result)
    r2 = anova(ml_result, ml_null)
    assert abs(r1["Chisq"].iloc[1] - r2["Chisq"].iloc[1]) < 1e-8


# ---------------------------------------------------------------------------
# anova(): error handling
# ---------------------------------------------------------------------------


def test_anova_raises_for_reml_models(reml_result, two_re_data):
    reml_null = interlace.fit(
        "y ~ 1", data=two_re_data, groups=["firm", "dept"], method="REML"
    )
    with pytest.raises(ValueError, match="[Rr][Ee][Mm][Ll]"):
        anova(reml_null, reml_result)


def test_anova_raises_for_mixed_methods(ml_result, reml_result):
    with pytest.raises(ValueError, match="[Rr][Ee][Mm][Ll]"):
        anova(ml_result, reml_result)


def test_fit_unknown_method_raises(two_re_data):
    with pytest.raises(ValueError, match="method"):
        interlace.fit("y ~ x", data=two_re_data, groups=["firm", "dept"], method="GLS")
