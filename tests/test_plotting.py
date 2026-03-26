"""Tests for interlace.plotting: plot_resid, plot_influence, dotplot_diag."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.influence import hlm_influence
from interlace.plotting import dotplot_diag, plot_influence, plot_resid
from interlace.residuals import hlm_resid


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
def il(data):
    return interlace.fit("y ~ x", data=data, groups="group")


@pytest.fixture(scope="module")
def resid_df(il):
    return hlm_resid(il, type="marginal", full_data=False)


@pytest.fixture(scope="module")
def influence_df(il):
    return hlm_influence(il, level=1)


def test_plot_resid_returns_ggplot(resid_df):
    p = plot_resid(resid_df)
    assert isinstance(p, ggplot)


def test_plot_resid_qq_returns_ggplot(resid_df):
    p = plot_resid(resid_df, type="qq")
    assert isinstance(p, ggplot)


def test_plot_resid_invalid_type_raises(resid_df):
    with pytest.raises(ValueError, match="type must be"):
        plot_resid(resid_df, type="invalid")


def test_plot_influence_returns_ggplot(influence_df):
    p = plot_influence(influence_df)
    assert isinstance(p, ggplot)


def test_dotplot_diag_returns_ggplot(influence_df):
    p = dotplot_diag(influence_df, diag="cooksd")
    assert isinstance(p, ggplot)


def test_dotplot_diag_with_numeric_cutoff(influence_df):
    p = dotplot_diag(influence_df, diag="cooksd", cutoff=0.1)
    assert isinstance(p, ggplot)
