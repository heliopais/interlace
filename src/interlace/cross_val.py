"""Cross-validation utilities with group-aware splits for mixed models.

Provides:
- cross_val(): LOGO and KFold CV respecting grouped data structure
- CVResult: dataclass with per-fold scores, mean, std, optional fold models
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import narwhals as nw
import numpy as np


@dataclass
class CVResult:
    """Result of a cross-validation run.

    Attributes
    ----------
    scores:
        Per-fold scores as a 1-D numpy array.
    fold_results:
        Optional list of per-fold dicts (returned when ``return_models=True``).
        Each dict contains ``"model"`` (fitted result), ``"train_groups"``,
        ``"test_groups"``, ``"y_true"``, and ``"y_pred"``.
    """

    scores: np.ndarray
    fold_results: list[dict[str, Any]] | None = field(default=None)

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores, ddof=1))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _resolve_scorer(scoring: str | Callable[..., float]) -> Callable[..., float]:
    if callable(scoring):
        return scoring
    if scoring == "rmse":
        return _rmse
    if scoring == "mae":
        return _mae
    raise ValueError(f"scoring must be 'rmse', 'mae', or a callable; got {scoring!r}")


def cross_val(
    formula: str,
    data: Any,
    groups: str,
    cv: str = "logo",
    k: int = 5,
    scoring: str | Callable[..., float] = "rmse",
    return_models: bool = False,
    **fit_kwargs: Any,
) -> CVResult:
    """Cross-validate a mixed model with group-aware splits.

    Parameters
    ----------
    formula:
        Fixed-effects formula, e.g. ``"y ~ x1 + x2"``.
    data:
        DataFrame (pandas, polars, or any narwhals-compatible frame).
    groups:
        Column name of the grouping variable (used both as the random effect
        grouping factor and to define CV folds).
    cv:
        ``"logo"`` for leave-one-group-out or ``"kfold"`` for k-fold.
    k:
        Number of folds for ``cv="kfold"``.
    scoring:
        ``"rmse"`` (default), ``"mae"``, or a callable
        ``scorer(y_true, y_pred) -> float``.
    return_models:
        If ``True``, each fold's fitted model and split metadata are stored in
        ``CVResult.fold_results``.
    **fit_kwargs:
        Extra keyword arguments forwarded to ``interlace.fit()``
        (e.g. ``random``, ``method``, ``optimizer``).

    Returns
    -------
    CVResult
    """
    if cv not in ("logo", "kfold"):
        raise ValueError(f"cv must be 'logo' or 'kfold'; got {cv!r}")

    scorer = _resolve_scorer(scoring)

    # Normalise to pandas for slicing (interlace.fit accepts pandas).
    nw_frame = nw.from_native(data, eager_only=True)
    try:
        import pandas as pd

        pdf = pd.DataFrame({col: nw_frame[col].to_numpy() for col in nw_frame.columns})
    except ImportError as exc:
        raise RuntimeError(  # pragma: no cover
            "cross_val requires pandas to be installed."
        ) from exc

    response = formula.split("~")[0].strip()
    unique_groups = sorted(pdf[groups].unique().tolist())

    if cv == "logo":
        folds = _logo_folds(pdf, groups, unique_groups)
    else:
        folds = _kfold_folds(pdf, groups, unique_groups, k)

    # Try tqdm, fall back to plain iteration.
    try:
        from tqdm.auto import tqdm

        fold_iter = tqdm(folds, desc=f"cross_val ({cv})", leave=False)
    except ImportError:
        fold_iter = folds  # type: ignore[assignment]

    # Lazy import to avoid circular dependency.
    import interlace as _il

    scores: list[float] = []
    fold_results: list[dict[str, Any]] | None = [] if return_models else None

    for train_df, test_df, train_groups, test_groups in fold_iter:
        model = _il.fit(formula, data=train_df, groups=groups, **fit_kwargs)
        y_pred = model.predict(test_df)
        y_true = test_df[response].to_numpy()
        score = scorer(y_true, y_pred)
        scores.append(score)

        if return_models and fold_results is not None:
            fold_results.append(
                {
                    "model": model,
                    "train_groups": train_groups,
                    "test_groups": test_groups,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )

    return CVResult(
        scores=np.array(scores),
        fold_results=fold_results if return_models else None,
    )


# ---------------------------------------------------------------------------
# Internal split generators
# ---------------------------------------------------------------------------


def _logo_folds(
    pdf: Any,
    groups_col: str,
    unique_groups: list[Any],
) -> list[tuple[Any, Any, list[Any], list[Any]]]:
    folds = []
    for g in unique_groups:
        mask = pdf[groups_col] == g
        train_df = pdf[~mask].reset_index(drop=True)
        test_df = pdf[mask].reset_index(drop=True)
        train_groups = sorted(train_df[groups_col].unique().tolist())
        test_groups = [g]
        folds.append((train_df, test_df, train_groups, test_groups))
    return folds


def _kfold_folds(
    pdf: Any,
    groups_col: str,
    unique_groups: list[Any],
    k: int,
) -> list[tuple[Any, Any, list[Any], list[Any]]]:
    """Assign groups (not observations) to folds."""
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(unique_groups))
    group_arr = np.array(unique_groups)
    fold_assignments = np.array_split(perm, k)

    folds = []
    for fold_idx in fold_assignments:
        test_groups = group_arr[fold_idx].tolist()
        train_groups = [g for g in unique_groups if g not in set(test_groups)]
        test_mask = pdf[groups_col].isin(test_groups)
        train_df = pdf[~test_mask].reset_index(drop=True)
        test_df = pdf[test_mask].reset_index(drop=True)
        folds.append((train_df, test_df, train_groups, test_groups))
    return folds
