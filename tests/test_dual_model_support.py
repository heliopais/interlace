"""Integration test: all diagnostic functions accept both CrossedLMEResult and
MixedLMResults."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.augment import hlm_augment
from interlace.influence import hlm_influence, n_influential, tau_gap
from interlace.leverage import leverage
from interlace.plotting import dotplot_diag, plot_influence, plot_resid
from interlace.residuals import hlm_resid


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_groups, n_per = 8, 5
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, n_groups)
    eps = rng.normal(0, 0.5, n)
    y = 1.0 + 0.8 * x + u[group_ids] + eps
    return pd.DataFrame({"y": y, "x": x, "group": group_ids.astype(str)})


@pytest.fixture(scope="module")
def sm(data):
    return MixedLM.from_formula("y ~ x", groups="group", data=data).fit(reml=True)


@pytest.fixture(scope="module")
def il(data):
    return interlace.fit("y ~ x", data=data, groups="group")


@pytest.mark.parametrize("model_fixture", ["sm", "il"])
def test_hlm_resid_both_types(model_fixture, sm, il):
    model = sm if model_fixture == "sm" else il
    result = hlm_resid(model, type="marginal", full_data=False)
    assert isinstance(result, pd.DataFrame)
    assert ".resid" in result.columns


@pytest.mark.parametrize("model_fixture", ["sm", "il"])
def test_leverage_both_types(model_fixture, sm, il):
    model = sm if model_fixture == "sm" else il
    result = leverage(model)
    assert isinstance(result, pd.DataFrame)
    assert "overall" in result.columns


@pytest.mark.parametrize("model_fixture", ["sm", "il"])
def test_hlm_influence_both_types(model_fixture, sm, il):
    model = sm if model_fixture == "sm" else il
    result = hlm_influence(model, level=1)
    assert isinstance(result, pd.DataFrame)
    assert "cooksd" in result.columns


@pytest.mark.parametrize("model_fixture", ["sm", "il"])
def test_n_influential_both_types(model_fixture, sm, il):
    model = sm if model_fixture == "sm" else il
    assert isinstance(n_influential(model), int)


@pytest.mark.parametrize("model_fixture", ["sm", "il"])
def test_tau_gap_both_types(model_fixture, sm, il):
    model = sm if model_fixture == "sm" else il
    result = tau_gap(model)
    assert isinstance(result, dict)


@pytest.mark.parametrize("model_fixture", ["sm", "il"])
def test_hlm_augment_both_types(model_fixture, sm, il):
    model = sm if model_fixture == "sm" else il
    result = hlm_augment(model, include_influence=False)
    assert isinstance(result, pd.DataFrame)
    assert ".resid" in result.columns


def test_plotting_functions_accept_diagnostic_output(il):
    resid = hlm_resid(il, type="marginal", full_data=False)
    infl = hlm_influence(il, level=1)
    assert isinstance(plot_resid(resid), ggplot)
    assert isinstance(plot_influence(infl), ggplot)
    assert isinstance(dotplot_diag(infl), ggplot)
