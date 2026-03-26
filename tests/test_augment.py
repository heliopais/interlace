"""Tests for interlace.augment.hlm_augment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.augment import hlm_augment


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_groups, n_per = 8, 4
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


def test_returns_dataframe(models):
    sm, il = models
    for model in (sm, il):
        result = hlm_augment(model)
        assert isinstance(result, pd.DataFrame)


def test_contains_resid_and_fitted(models):
    sm, il = models
    for model in (sm, il):
        result = hlm_augment(model)
        assert ".resid" in result.columns
        assert ".fitted" in result.columns


def test_contains_influence_when_requested(models):
    sm, il = models
    for model in (sm, il):
        result = hlm_augment(model, include_influence=True)
        assert "cooksd" in result.columns


def test_no_influence_when_not_requested(models):
    sm, il = models
    for model in (sm, il):
        result = hlm_augment(model, include_influence=False)
        assert "cooksd" not in result.columns


def test_length_equals_nobs(models, data):
    sm, il = models
    for model in (sm, il):
        result = hlm_augment(model)
        assert len(result) == len(data)
