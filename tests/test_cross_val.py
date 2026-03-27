"""Tests for cross_val: LOGO and KFold CV utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interlace.cross_val import CVResult, cross_val

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def sleepstudy() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "lme4_sleepstudy_data.csv")


# ---------------------------------------------------------------------------
# CVResult dataclass
# ---------------------------------------------------------------------------


def test_cvresult_attributes():
    scores = np.array([1.0, 2.0, 3.0])
    result = CVResult(scores=scores, fold_results=None)
    assert result.mean == pytest.approx(2.0)
    assert result.std == pytest.approx(np.std(scores, ddof=1))
    assert result.fold_results is None


# ---------------------------------------------------------------------------
# LOGO CV
# ---------------------------------------------------------------------------


def test_logo_returns_cvresult(sleepstudy):
    cv = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="logo",
    )
    assert isinstance(cv, CVResult)
    # sleepstudy has 18 subjects → 18 folds
    assert len(cv.scores) == 18


def test_logo_rmse_reasonable(sleepstudy):
    """LOGO RMSE on sleepstudy should be ~40-80 ms (lme4 community benchmarks)."""
    cv = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="logo",
        scoring="rmse",
    )
    assert 30.0 < cv.mean < 100.0, f"LOGO RMSE={cv.mean:.2f} out of expected range"


def test_logo_mae_scoring(sleepstudy):
    cv = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="logo",
        scoring="mae",
    )
    assert cv.mean > 0
    assert len(cv.scores) == 18


def test_logo_custom_scorer(sleepstudy):
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    cv = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="logo",
        scoring=mse,
    )
    assert cv.mean > 0


def test_logo_fold_results_optional(sleepstudy):
    cv_no = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="logo",
        return_models=False,
    )
    assert cv_no.fold_results is None

    cv_yes = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="logo",
        return_models=True,
    )
    assert cv_yes.fold_results is not None
    assert len(cv_yes.fold_results) == 18


# ---------------------------------------------------------------------------
# KFold CV
# ---------------------------------------------------------------------------


def test_kfold_no_group_leakage(sleepstudy):
    """No subject should appear in both train and test for any fold."""
    cv = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="kfold",
        k=5,
        return_models=True,
    )
    assert cv.fold_results is not None
    for fold in cv.fold_results:
        train_groups = set(fold["train_groups"])
        test_groups = set(fold["test_groups"])
        assert train_groups.isdisjoint(test_groups), (
            f"Group leakage: {train_groups & test_groups}"
        )


def test_kfold_k_folds(sleepstudy):
    cv = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="kfold",
        k=5,
    )
    assert len(cv.scores) == 5


def test_kfold_rmse_reasonable(sleepstudy):
    cv = cross_val(
        "Reaction ~ Days",
        data=sleepstudy,
        groups="Subject",
        cv="kfold",
        k=3,
        scoring="rmse",
    )
    assert 20.0 < cv.mean < 150.0, f"KFold RMSE={cv.mean:.2f} out of expected range"


# ---------------------------------------------------------------------------
# Narwhals / polars compatibility
# ---------------------------------------------------------------------------


def test_logo_polars(sleepstudy):
    pytest.importorskip("polars")
    import polars as pl

    df_polars = pl.from_pandas(sleepstudy)
    cv = cross_val(
        "Reaction ~ Days",
        data=df_polars,
        groups="Subject",
        cv="logo",
        scoring="rmse",
    )
    assert isinstance(cv, CVResult)
    assert len(cv.scores) == 18
    assert cv.mean > 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_cv_mode_raises(sleepstudy):
    with pytest.raises(ValueError, match="cv must be"):
        cross_val(
            "Reaction ~ Days",
            data=sleepstudy,
            groups="Subject",
            cv="bad_mode",
        )


def test_invalid_scoring_raises(sleepstudy):
    with pytest.raises(ValueError, match="scoring must be"):
        cross_val(
            "Reaction ~ Days",
            data=sleepstudy,
            groups="Subject",
            cv="logo",
            scoring="r2",
        )
