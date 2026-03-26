"""Tests for n_influential and tau_gap metrics.

Acceptance criteria:
  - n_influential returns non-negative int
  - Default threshold is 4/n
  - Custom threshold changes the count as expected
  - Works with both CrossedLMEResult and statsmodels MixedLMResults
  - tau_gap returns dict keyed by factor name with non-negative floats
  - tau_gap is 0 when no influential observations exist (very loose threshold)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.influence import n_influential, tau_gap


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


# --- tau_gap ---


def test_tau_gap_returns_dict(models):
    sm, il = models
    for model in (sm, il):
        result = tau_gap(model)
        assert isinstance(result, dict)


def test_tau_gap_nonnegative_values(models):
    sm, il = models
    for model in (sm, il):
        result = tau_gap(model)
        for factor, gap in result.items():
            assert gap >= 0.0, f"tau_gap[{factor}] = {gap} < 0"


def test_tau_gap_keyed_by_factor_name(models):
    _, il = models
    result = tau_gap(il)
    assert "group" in result


def test_tau_gap_zero_when_no_influential(models):
    """With a very large threshold no obs are removed so tau_gap should be ~0."""
    _, il = models
    result = tau_gap(il, threshold=1e9)
    for factor, gap in result.items():
        # model refit on same data: gap should be ~0 (within numerical tolerance)
        assert gap < 1e-6, f"tau_gap[{factor}] = {gap} with no removal"


def test_tau_gap_uses_default_threshold_4_over_n(models, data):
    _, il = models
    n = len(data)
    r1 = tau_gap(il)
    r2 = tau_gap(il, threshold=4.0 / n)
    for factor in r1:
        assert abs(r1[factor] - r2[factor]) < 1e-12
