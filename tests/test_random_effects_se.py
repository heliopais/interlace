"""Tests for random_effects_se and random_effects_ci on CrossedLMEResult.

Uses:
  - Dyestuff  (1 | Batch)       — intercept-only, balanced: SE has closed form
  - sleepstudy (Days | Subject)  — correlated random slopes: structure checks only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def dyestuff() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "lme4_dyestuff_data.csv")


@pytest.fixture
def sleepstudy() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "lme4_sleepstudy_data.csv")


@pytest.fixture
def dyestuff_fit(dyestuff):
    return interlace.fit("Yield ~ 1", dyestuff, groups="Batch")


@pytest.fixture
def sleepstudy_fit(sleepstudy):
    return interlace.fit("Reaction ~ Days", sleepstudy, random=["(1 + Days | Subject)"])


# ---------------------------------------------------------------------------
# random_effects_se
# ---------------------------------------------------------------------------


def test_random_effects_se_returns_dict(dyestuff_fit):
    se = dyestuff_fit.random_effects_se
    assert isinstance(se, dict)


def test_random_effects_se_same_keys_as_random_effects(dyestuff_fit):
    se = dyestuff_fit.random_effects_se
    assert set(se.keys()) == set(dyestuff_fit.random_effects.keys())


def test_random_effects_se_same_index_as_random_effects(dyestuff_fit):
    se = dyestuff_fit.random_effects_se
    re = dyestuff_fit.random_effects
    for group in se:
        np.testing.assert_array_equal(list(se[group].index), list(re[group].index))


def test_random_effects_se_all_positive_finite(dyestuff_fit):
    se = dyestuff_fit.random_effects_se
    for group, series in se.items():
        vals = np.asarray(series)
        assert np.all(vals > 0), f"Non-positive SE in {group}"
        assert np.all(np.isfinite(vals)), f"Non-finite SE in {group}"


def test_random_effects_se_dyestuff_analytical(dyestuff_fit):
    """For balanced intercept-only model, SE has a closed form.

    Dyestuff: 6 batches × 5 obs.  A11 = (5*θ² + 1)*I_6, so
    Var(b_j | y) = σ² * θ² / (5*θ² + 1)  for all j.
    """
    result = dyestuff_fit
    theta = float(result.theta[0])
    sigma2 = result.scale
    n_per_batch = 5

    expected_se = np.sqrt(sigma2 * theta**2 / (n_per_batch * theta**2 + 1))

    se = result.random_effects_se
    se_vals = np.asarray(se["Batch"])

    # All SEs should be identical (balanced data)
    assert np.allclose(se_vals, se_vals[0], rtol=1e-6)
    # Match closed form to 0.1% relative tolerance
    np.testing.assert_allclose(se_vals, expected_se, rtol=1e-3)


def test_random_effects_se_sleepstudy_structure(sleepstudy_fit):
    """Multi-term (random slopes) SE is a DataFrame matching random_effects."""
    se = sleepstudy_fit.random_effects_se
    re = sleepstudy_fit.random_effects

    assert "Subject" in se
    # Should be a DataFrame with same columns and index
    assert list(se["Subject"].columns) == list(re["Subject"].columns)
    assert list(se["Subject"].index) == list(re["Subject"].index)


def test_random_effects_se_sleepstudy_all_positive(sleepstudy_fit):
    se = sleepstudy_fit.random_effects_se
    vals = se["Subject"].to_numpy()
    assert np.all(vals > 0)
    assert np.all(np.isfinite(vals))


# ---------------------------------------------------------------------------
# random_effects_ci
# ---------------------------------------------------------------------------


def test_random_effects_ci_returns_dict(dyestuff_fit):
    ci = dyestuff_fit.random_effects_ci()
    assert isinstance(ci, dict)


def test_random_effects_ci_same_keys(dyestuff_fit):
    ci = dyestuff_fit.random_effects_ci()
    assert set(ci.keys()) == set(dyestuff_fit.random_effects.keys())


def test_random_effects_ci_columns(dyestuff_fit):
    ci = dyestuff_fit.random_effects_ci()
    for _group, df in ci.items():
        assert "lower" in df.columns
        assert "upper" in df.columns


def test_random_effects_ci_lower_lt_upper(dyestuff_fit):
    ci = dyestuff_fit.random_effects_ci()
    for _group, df in ci.items():
        assert np.all(df["lower"].to_numpy() < df["upper"].to_numpy())


def test_random_effects_ci_contains_blup(dyestuff_fit):
    """The point BLUP should lie strictly inside the 95% CI."""
    ci = dyestuff_fit.random_effects_ci(level=0.95)
    re = dyestuff_fit.random_effects
    for group in ci:
        blup_vals = np.asarray(re[group])
        lower = ci[group]["lower"].to_numpy()
        upper = ci[group]["upper"].to_numpy()
        assert np.all(lower < blup_vals) and np.all(blup_vals < upper)


def test_random_effects_ci_wider_for_higher_level(dyestuff_fit):
    """99% CI must be strictly wider than 90% CI for every group."""
    ci90 = dyestuff_fit.random_effects_ci(level=0.90)
    ci99 = dyestuff_fit.random_effects_ci(level=0.99)
    for group in ci90:
        width90 = ci90[group]["upper"].to_numpy() - ci90[group]["lower"].to_numpy()
        width99 = ci99[group]["upper"].to_numpy() - ci99[group]["lower"].to_numpy()
        assert np.all(width99 > width90)


def test_random_effects_ci_symmetric(dyestuff_fit):
    """CIs should be symmetric around the BLUP (normal approximation)."""
    ci = dyestuff_fit.random_effects_ci()
    re = dyestuff_fit.random_effects
    for group in ci:
        blup = np.asarray(re[group])
        mid = (ci[group]["lower"].to_numpy() + ci[group]["upper"].to_numpy()) / 2
        np.testing.assert_allclose(mid, blup, atol=1e-10)


def test_random_effects_ci_sleepstudy_multi_term(sleepstudy_fit):
    """CI for multi-term RE returns a dict of DataFrames with correct structure."""
    ci = sleepstudy_fit.random_effects_ci()
    assert "Subject" in ci
    df = ci["Subject"]
    # Should have MultiIndex or two-level columns (term × bound)
    # or a DataFrame with columns [lower, upper] per term
    # Depending on implementation: at minimum lower < upper for all entries
    assert "lower" in df.columns or isinstance(df.columns, pd.MultiIndex)
