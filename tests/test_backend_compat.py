"""Regression tests: fit-result attributes are always pandas regardless of input.

Augmented/diagnostic frames (augment, residuals, leverage, influence) follow
the input backend via _frame.to_native().  Fit-result attributes (fe_params,
fe_bse, fe_pvalues, fe_df, fe_conf_int, random_effects) are always pd.Series /
pd.DataFrame because polars Series have no named-index concept and gpgap
depends on the statsmodels-compatible pandas API.

These are contract tests — they confirm documented behaviour rather than
driving new implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def polars_result():
    """Small LMM fitted from a polars DataFrame."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    n_groups, n_per = 10, 6
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, n_groups)
    y = 2.0 + 0.5 * x + u[group] + rng.normal(0, 0.5, n)
    df_pl = pl.DataFrame({"y": y, "x": x, "g": group.astype(str)})
    return interlace.fit("y ~ x", df_pl, groups="g")


# ---------------------------------------------------------------------------
# Fit-result attrs: always pandas
# ---------------------------------------------------------------------------


def test_fe_params_always_pandas(polars_result):
    assert isinstance(polars_result.fe_params, pd.Series)


def test_fe_bse_always_pandas(polars_result):
    assert isinstance(polars_result.fe_bse, pd.Series)


def test_fe_pvalues_always_pandas(polars_result):
    assert isinstance(polars_result.fe_pvalues, pd.Series)


def test_fe_df_always_pandas(polars_result):
    assert isinstance(polars_result.fe_df, pd.Series)


def test_fe_conf_int_always_pandas(polars_result):
    assert isinstance(polars_result.fe_conf_int, pd.DataFrame)


def test_random_effects_always_pandas(polars_result):
    for v in polars_result.random_effects.values():
        assert isinstance(v, pd.Series | pd.DataFrame)


# ---------------------------------------------------------------------------
# Augmented frames: follow the input backend
# ---------------------------------------------------------------------------


def test_augment_follows_polars_input(polars_result):
    pl = pytest.importorskip("polars")
    aug = interlace.hlm_augment(polars_result)
    assert isinstance(aug, pl.DataFrame)


def test_residuals_follows_polars_input(polars_result):
    pl = pytest.importorskip("polars")
    res = interlace.hlm_resid(polars_result)
    assert isinstance(res, pl.DataFrame)
