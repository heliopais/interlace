"""Tests for interlace.leverage.leverage.

Acceptance criteria:
  - Returns DataFrame with overall, fixef, ranef, ranef.uc columns
  - Works with both CrossedLMEResult and statsmodels MixedLMResults
  - fixef values from both correlate > 0.99
  - overall values from both correlate > 0.99
  - overall ≈ fixef + ranef (within floating-point tolerance)
  - All leverage values are non-negative
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.leverage import leverage


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_groups, n_per = 20, 10
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.2, n_groups)
    eps = rng.normal(0, 0.8, n)
    y = 2.0 + 1.5 * x + u[group_ids] + eps
    return pd.DataFrame({"y": y, "x": x, "group": group_ids.astype(str)})


@pytest.fixture(scope="module")
def models(data):
    sm = MixedLM.from_formula("y ~ x", groups="group", data=data).fit(reml=True)
    il = interlace.fit("y ~ x", data=data, groups="group")
    return sm, il


# --- structure ---


def test_returns_dataframe_with_required_cols(models):
    sm, il = models
    for model in (sm, il):
        result = leverage(model)
        assert isinstance(result, pd.DataFrame)
        for col in ("overall", "fixef", "ranef", "ranef.uc"):
            assert col in result.columns, f"missing column: {col}"


def test_length_equals_nobs(models, data):
    sm, il = models
    for model in (sm, il):
        result = leverage(model)
        assert len(result) == len(data)


def test_values_nonnegative(models):
    sm, il = models
    for model in (sm, il):
        result = leverage(model)
        for col in ("overall", "fixef", "ranef", "ranef.uc"):
            assert (result[col] >= -1e-10).all(), f"negative values in {col}"


def test_overall_equals_fixef_plus_ranef(models):
    sm, il = models
    for model in (sm, il):
        result = leverage(model)
        np.testing.assert_allclose(
            result["overall"].values,
            result["fixef"].values + result["ranef"].values,
            atol=1e-10,
        )


# --- parity across model types ---


def test_fixef_leverage_match_across_models(models):
    sm, il = models
    sm_h1 = leverage(sm)["fixef"].values
    il_h1 = leverage(il)["fixef"].values
    corr = np.corrcoef(sm_h1, il_h1)[0, 1]
    assert corr > 0.99, f"fixef leverage correlation={corr:.4f}"


def test_overall_leverage_match_across_models(models):
    sm, il = models
    sm_h = leverage(sm)["overall"].values
    il_h = leverage(il)["overall"].values
    corr = np.corrcoef(sm_h, il_h)[0, 1]
    assert corr > 0.99, f"overall leverage correlation={corr:.4f}"
