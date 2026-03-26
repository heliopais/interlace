"""Residual extraction for fitted linear mixed models.

Supports both ``CrossedLMEResult`` and statsmodels ``MixedLMResults``.

Note on statsmodels: ``MixedLMResults.fittedvalues`` includes predicted random
effects (conditional fitted values), so ``MixedLMResults.resid`` is the
conditional residual ``y - Xβ - Zu``.  Marginal residuals (``y - Xβ``) must be
computed explicitly from the stored endog and fixed-effects design matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from interlace._frame import to_native as _to_native
from interlace.result import CrossedLMEResult


def _is_crossed(model: Any) -> bool:
    return isinstance(model, CrossedLMEResult)


def _marginal(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(marginal_resid, marginal_fitted)`` = ``(y - Xβ, Xβ)``."""
    # Works for both CrossedLMEResult and statsmodels MixedLMResults since both
    # expose model.model.exog, model.model.endog, and model.fe_params.
    xbeta = np.asarray(model.model.exog @ model.fe_params.values, dtype=float)
    endog = np.asarray(model.model.endog, dtype=float)
    return endog - xbeta, xbeta


def _conditional(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(conditional_resid, conditional_fitted)`` = ``(y - Xβ - Zu, Xβ + Zu)``."""  # noqa: E501
    # Both CrossedLMEResult and statsmodels MixedLMResults store conditional
    # residuals / fitted values in .resid / .fittedvalues.
    return np.asarray(model.resid, dtype=float), np.asarray(
        model.fittedvalues, dtype=float
    )


def hlm_resid(
    model: Any,
    full_data: bool = True,
    type: str = "marginal",  # noqa: A002
    standardized: bool = False,
    level: int | str = 1,
) -> Any:
    """Extract residuals from a fitted linear mixed model.

    Parameters
    ----------
    model:
        A ``CrossedLMEResult`` or statsmodels ``MixedLMResults`` object.
    full_data:
        If ``True``, prepend the original data columns to the result.
    type:
        ``"marginal"`` (ignoring random effects, ``y - Xβ``) or
        ``"conditional"`` (subtracting predicted random effects, ``y - Xβ - Zu``).
        Only used when ``level=1``.
    standardized:
        If ``True``, divide residuals by ``sqrt(scale)``.
    level:
        ``1`` for observation-level residuals; a factor/group name for
        group-level random effects.

    Returns
    -------
    pd.DataFrame
    """
    if level == 1:
        if type not in ("marginal", "conditional"):
            msg = "type must be 'marginal' or 'conditional'"
            raise ValueError(msg)

        res, fitted = _marginal(model) if type == "marginal" else _conditional(model)

        if standardized:
            res = res / np.sqrt(model.scale)

        res_df = pd.DataFrame({".resid": res, ".fitted": fitted})

        native_frame = model.model.data.frame
        if full_data:
            # For CrossedLMEResult, _pandas_frame holds the cached pandas copy.
            # For statsmodels, .frame is already a pandas DataFrame.
            pd_frame = getattr(model.model.data, "_pandas_frame", None)
            if pd_frame is None:
                pd_frame = native_frame
            result_df = pd.concat(
                [pd_frame.reset_index(drop=True), res_df],
                axis=1,
            )
            return _to_native(result_df, like=native_frame)

        return _to_native(res_df, like=native_frame)

    # --- group-level: return random effects ---
    if _is_crossed(model):
        if level not in model.random_effects:
            available = list(model.random_effects)
            msg = f"level '{level}' not found in random_effects; available: {available}"
            raise ValueError(msg)
        re_series = model.random_effects[level]
        re_df = pd.DataFrame(
            {f".ranef.{re_series.name}": re_series.values},
            index=re_series.index,
        )
        re_df.index.name = level
        return re_df.reset_index()

    # statsmodels: random_effects is {group_label: Series}
    re = model.random_effects
    re_df = pd.DataFrame.from_dict(re, orient="index")
    re_df.index.name = level
    re_df.columns = [f".ranef.{col}" for col in re_df.columns]
    return re_df.reset_index()
