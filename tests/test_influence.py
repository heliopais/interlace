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
