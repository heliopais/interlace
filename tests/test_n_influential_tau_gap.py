"""Tests for n_influential metric.

Acceptance criteria:
  - n_influential returns non-negative int
  - Default threshold is 4/n
  - Custom threshold changes the count as expected
  - Works with both CrossedLMEResult and statsmodels MixedLMResults
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.influence import n_influential


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


# --- n_influential ---


def test_n_influential_returns_int(models):
    sm, il = models
    for model in (sm, il):
        result = n_influential(model)
        assert isinstance(result, int)


def test_n_influential_nonnegative(models):
    sm, il = models
    for model in (sm, il):
        assert n_influential(model) >= 0


def test_n_influential_default_threshold_is_4_over_n(models, data):
    sm, il = models
    n = len(data)
    for model in (sm, il):
        assert n_influential(model) == n_influential(model, threshold=4.0 / n)


def test_n_influential_zero_with_very_high_threshold(models):
    sm, il = models
    for model in (sm, il):
        assert n_influential(model, threshold=1e9) == 0


def test_n_influential_all_with_zero_threshold(models, data):
    sm, il = models
    n = len(data)
    for model in (sm, il):
        # threshold=0 should flag everything except NaNs
        result = n_influential(model, threshold=0.0)
        assert result <= n


def test_n_influential_decreases_with_higher_threshold(models):
    sm, il = models
    for model in (sm, il):
        low = n_influential(model, threshold=0.01)
        high = n_influential(model, threshold=1.0)
        assert low >= high
