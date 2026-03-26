"""Parity tests: polars DataFrame input produces identical results to pandas.

These tests define the acceptance criteria for Phase 1 narwhals support.
They are expected to FAIL until the narwhals API boundary is implemented.

Acceptance criteria:
- fit() accepts pl.DataFrame without error
- Numerical outputs (fe_params, scale, theta, BLUPs, residuals) are identical
  to within the same tolerances used for R/lme4 parity
- model.data.frame returns pl.DataFrame when polars input was given
- hlm_resid / hlm_augment return pl.DataFrame for polars-fitted models
- predict(newdata=pl.DataFrame) works without error
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

polars = pytest.importorskip("polars")

import interlace  # noqa: E402
from interlace.augment import hlm_augment  # noqa: E402
from interlace.residuals import hlm_resid  # noqa: E402


@pytest.fixture(scope="module")
def base_data() -> pd.DataFrame:
    """Synthetic dataset: y ~ x + (1|group), 15 groups × 8 obs."""
    rng = np.random.default_rng(7)
    n_groups, n_per = 15, 8
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.2, n_groups)
    eps = rng.normal(0, 0.8, n)
    y = 2.0 + 1.5 * x + u[group_ids] + eps
    return pd.DataFrame({"y": y, "x": x, "group": group_ids.astype(str)})


@pytest.fixture(scope="module")
def pd_result(base_data):
    return interlace.fit("y ~ x", data=base_data, groups="group")


@pytest.fixture(scope="module")
def pl_data(base_data) -> polars.DataFrame:
    return polars.DataFrame(
        {col: base_data[col].to_numpy() for col in base_data.columns}
    )


@pytest.fixture(scope="module")
def pl_result(pl_data):
    return interlace.fit("y ~ x", data=pl_data, groups="group")


# ---------------------------------------------------------------------------
# fit() acceptance
# ---------------------------------------------------------------------------


def test_fit_accepts_polars(pl_data):
    """fit() must not raise when given a polars DataFrame."""
    result = interlace.fit("y ~ x", data=pl_data, groups="group")
    assert result is not None


# ---------------------------------------------------------------------------
# Numerical parity
# ---------------------------------------------------------------------------


def test_fixed_effects_parity(pd_result, pl_result):
    for name in pd_result.fe_params.index:
        diff = abs(pl_result.fe_params[name] - pd_result.fe_params[name])
        assert diff < 1e-10, f"fe_params['{name}'] differ: {diff:.2e}"


def test_scale_parity(pd_result, pl_result):
    diff = abs(pl_result.scale - pd_result.scale)
    assert diff < 1e-10, f"scale differs: {diff:.2e}"


def test_theta_parity(pd_result, pl_result):
    np.testing.assert_allclose(pl_result.theta, pd_result.theta, atol=1e-10)


def test_blup_parity(base_data, pd_result, pl_result):
    group_labels = sorted(base_data["group"].unique())
    pd_blups = np.array([pd_result.random_effects["group"][g] for g in group_labels])
    pl_blups = np.array([pl_result.random_effects["group"][g] for g in group_labels])
    np.testing.assert_allclose(pl_blups, pd_blups, atol=1e-10)


def test_residuals_parity(pd_result, pl_result):
    np.testing.assert_allclose(
        np.asarray(pl_result.resid), np.asarray(pd_result.resid), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Output type contract
# ---------------------------------------------------------------------------


def test_data_frame_is_polars(pl_result):
    """model.data.frame must return pl.DataFrame when polars was passed to fit()."""
    assert isinstance(pl_result.model.data.frame, polars.DataFrame)


def test_hlm_resid_returns_polars(pl_result):
    result = hlm_resid(pl_result, type="marginal")
    assert isinstance(result, polars.DataFrame)


def test_hlm_resid_conditional_returns_polars(pl_result):
    result = hlm_resid(pl_result, type="conditional")
    assert isinstance(result, polars.DataFrame)


def test_hlm_augment_returns_polars(pl_result):
    result = hlm_augment(pl_result, include_influence=False)
    assert isinstance(result, polars.DataFrame)


def test_predict_with_polars_newdata(pl_result, pl_data):
    """predict() must accept a polars DataFrame as newdata."""
    preds = pl_result.predict(newdata=pl_data)
    assert len(preds) == len(pl_data)
