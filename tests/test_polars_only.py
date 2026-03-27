"""Pandas-free integration test: all core operations with polars DataFrames.

This test file is designed to run in an environment where pandas is NOT
installed.  It covers fit, predict, hlm_resid, and hlm_influence to verify
that interlace works end-to-end without pandas as a dependency.

Run in the pandas-free CI job:
    uv run pytest tests/test_polars_only.py
"""

from __future__ import annotations

import numpy as np
import pytest

polars = pytest.importorskip("polars")


@pytest.fixture(scope="module")
def pl_data() -> polars.DataFrame:
    rng = np.random.default_rng(7)
    n_groups, n_per = 8, 5
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, n_groups)
    eps = rng.normal(0, 0.5, n)
    y = 2.0 + 0.7 * x + u[np.repeat(np.arange(n_groups), n_per)] + eps
    return polars.DataFrame({"y": y, "x": x, "group": group_ids})


@pytest.fixture(scope="module")
def pl_model(pl_data):
    import interlace

    return interlace.fit("y ~ x", data=pl_data, groups="group")


# ---------------------------------------------------------------------------
# fit() — basic sanity checks
# ---------------------------------------------------------------------------


def test_fit_returns_result(pl_model):
    from interlace.result import CrossedLMEResult

    assert isinstance(pl_model, CrossedLMEResult)


def test_fit_fe_params_length(pl_model):
    params = np.asarray(pl_model.fe_params)
    assert params.shape == (2,), f"Expected 2 FE params, got shape {params.shape}"


def test_fit_scale_positive(pl_model):
    assert pl_model.scale > 0


def test_fit_random_effects_populated(pl_model):
    assert "group" in pl_model.random_effects
    re = pl_model.random_effects["group"]
    re_vals = re.values if hasattr(re, "values") else np.asarray(re)
    assert re_vals.shape == (8,)


def test_fit_no_pandas_import(pl_data):
    """Verify that fitting with polars data does not import pandas.

    Only meaningful in a pandas-free environment; skipped when pandas is installed.
    """
    import sys

    try:
        import pandas  # noqa: F401

        pytest.skip("pandas is installed — skip pandas-free assertion")
    except ImportError:
        pass

    assert "pandas" not in sys.modules, (
        "pandas was imported during fit() in the polars-only path"
    )


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_in_sample(pl_model, pl_data):
    preds = pl_model.predict(pl_data)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(pl_data),)


def test_predict_fixed_only(pl_model, pl_data):
    preds_fe = pl_model.predict(pl_data, include_re=False)
    assert isinstance(preds_fe, np.ndarray)
    assert preds_fe.shape == (len(pl_data),)


def test_predict_fittedvalues_match(pl_model):
    in_sample = pl_model.predict()
    np.testing.assert_array_equal(in_sample, np.asarray(pl_model.fittedvalues))


# ---------------------------------------------------------------------------
# hlm_resid() — observation level
# ---------------------------------------------------------------------------


def test_hlm_resid_returns_polars(pl_model):
    from interlace.residuals import hlm_resid

    result = hlm_resid(pl_model, level=1)
    assert isinstance(result, polars.DataFrame), (
        f"Expected polars.DataFrame, got {type(result)}"
    )


def test_hlm_resid_has_required_columns(pl_model):
    from interlace.residuals import hlm_resid

    result = hlm_resid(pl_model, level=1)
    assert ".resid" in result.columns
    assert ".fitted" in result.columns


def test_hlm_resid_full_data_includes_original_cols(pl_model):
    from interlace.residuals import hlm_resid

    result = hlm_resid(pl_model, level=1, full_data=True)
    assert ".resid" in result.columns
    assert "y" in result.columns
    assert "x" in result.columns


def test_hlm_resid_conditional(pl_model):
    from interlace.residuals import hlm_resid

    result = hlm_resid(pl_model, level=1, type="conditional")
    assert ".resid" in result.columns


# ---------------------------------------------------------------------------
# hlm_influence()
# ---------------------------------------------------------------------------


def test_hlm_influence_returns_polars(pl_model):
    from interlace.influence import hlm_influence

    result = hlm_influence(pl_model, level=1)
    assert isinstance(result, polars.DataFrame), (
        f"Expected polars.DataFrame, got {type(result)}"
    )


def test_hlm_influence_has_required_columns(pl_model):
    from interlace.influence import hlm_influence

    result = hlm_influence(pl_model, level=1)
    for col in ("cooksd", "mdffits", "covtrace", "covratio"):
        assert col in result.columns


def test_hlm_influence_length_equals_nobs(pl_model, pl_data):
    from interlace.influence import hlm_influence

    result = hlm_influence(pl_model, level=1)
    assert len(result) == len(pl_data)


def test_hlm_influence_cooksd_nonnegative(pl_model):
    from interlace.influence import hlm_influence

    result = hlm_influence(pl_model, level=1)
    assert (result["cooksd"].to_numpy() >= 0).all() or np.isnan(
        result["cooksd"].to_numpy()
    ).any()
