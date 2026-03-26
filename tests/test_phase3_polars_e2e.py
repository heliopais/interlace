"""Phase 3 acceptance tests: end-to-end diagnostics with polars input.

TDD gate: these tests must pass after Phase 3 refactoring is complete.
Acceptance criteria (interlace-lbi.4):
- hlm_influence returns pl.DataFrame when model is fitted on polars input
- hlm_augment(include_influence=True) returns pl.DataFrame for polars-fitted models
- Numerical values match the equivalent pandas run
- All existing pandas-based tests continue to pass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

polars = pytest.importorskip("polars")

import interlace  # noqa: E402
from interlace.augment import hlm_augment  # noqa: E402
from interlace.influence import hlm_influence  # noqa: E402


@pytest.fixture(scope="module")
def base_data() -> pd.DataFrame:
    """Small synthetic dataset: y ~ x + (1|group), 6 groups × 4 obs."""
    rng = np.random.default_rng(42)
    n_groups, n_per = 6, 4
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, n_groups)
    eps = rng.normal(0, 0.5, n)
    y = 1.5 + 0.8 * x + u[group_ids] + eps
    return pd.DataFrame({"y": y, "x": x, "group": group_ids.astype(str)})


@pytest.fixture(scope="module")
def pd_model(base_data):
    return interlace.fit("y ~ x", data=base_data, groups="group")


@pytest.fixture(scope="module")
def pl_data(base_data) -> polars.DataFrame:
    return polars.DataFrame(
        {col: base_data[col].to_numpy() for col in base_data.columns}
    )


@pytest.fixture(scope="module")
def pl_model(pl_data):
    return interlace.fit("y ~ x", data=pl_data, groups="group")


# ---------------------------------------------------------------------------
# hlm_influence with polars input
# ---------------------------------------------------------------------------


def test_hlm_influence_returns_polars(pl_model):
    """hlm_influence must return pl.DataFrame when model was fitted on polars."""
    result = hlm_influence(pl_model, level=1)
    assert isinstance(result, polars.DataFrame), (
        f"Expected polars.DataFrame, got {type(result)}"
    )


def test_hlm_influence_has_expected_columns(pl_model):
    result = hlm_influence(pl_model, level=1)
    for col in ("cooksd", "mdffits", "covtrace", "covratio"):
        assert col in result.columns, f"Missing column: {col}"


def test_hlm_influence_numerical_parity(pd_model, pl_model):
    """Influence stats from polars model must match pandas model numerically."""
    pd_infl = hlm_influence(pd_model, level=1)
    pl_infl = hlm_influence(pl_model, level=1)

    for col in ("cooksd", "mdffits", "covtrace", "covratio"):
        np.testing.assert_allclose(
            pl_infl[col].to_numpy(),
            pd_infl[col].to_numpy(),
            atol=1e-10,
            err_msg=f"Column '{col}' differs between polars and pandas models",
        )


# ---------------------------------------------------------------------------
# hlm_augment(include_influence=True) with polars input
# ---------------------------------------------------------------------------


def test_hlm_augment_with_influence_returns_polars(pl_model):
    """hlm_augment(include_influence=True) must return pl.DataFrame."""
    result = hlm_augment(pl_model, include_influence=True)
    assert isinstance(result, polars.DataFrame), (
        f"Expected polars.DataFrame, got {type(result)}"
    )


def test_hlm_augment_with_influence_has_expected_columns(pl_model):
    result = hlm_augment(pl_model, include_influence=True)
    for col in (".resid", ".fitted", "cooksd"):
        assert col in result.columns, f"Missing column: {col}"


def test_hlm_augment_with_influence_numerical_parity(pd_model, pl_model, base_data):
    """Augmented DataFrame from polars model must match pandas model."""
    pd_aug = hlm_augment(pd_model, include_influence=True)
    pl_aug = hlm_augment(pl_model, include_influence=True)

    for col in (".resid", ".fitted", "cooksd"):
        np.testing.assert_allclose(
            pl_aug[col].to_numpy(),
            np.asarray(pd_aug[col]),
            atol=1e-10,
            err_msg=f"Column '{col}' differs between polars and pandas augmented frames",  # noqa: E501
        )
