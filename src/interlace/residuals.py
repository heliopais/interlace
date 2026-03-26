"""Residual extraction for fitted linear mixed models.

Supports both ``CrossedLMEResult`` and statsmodels ``MixedLMResults``.

Note on statsmodels: ``MixedLMResults.fittedvalues`` includes predicted random
effects (conditional fitted values), so ``MixedLMResults.resid`` is the
conditional residual ``y - Xβ - Zu``.  Marginal residuals (``y - Xβ``) must be
computed explicitly from the stored endog and fixed-effects design matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from interlace.result import CrossedLMEResult


def _is_crossed(model) -> bool:
    return isinstance(model, CrossedLMEResult)


def _marginal(model) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(marginal_resid, marginal_fitted)`` = ``(y - Xβ, Xβ)``."""
    # Works for both CrossedLMEResult and statsmodels MixedLMResults since both
    # expose model.model.exog, model.model.endog, and model.fe_params.
    xbeta = np.asarray(model.model.exog @ model.fe_params.values, dtype=float)
    endog = np.asarray(model.model.endog, dtype=float)
    return endog - xbeta, xbeta


def _conditional(model) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(conditional_resid, conditional_fitted)`` = ``(y - Xβ - Zu, Xβ + Zu)``."""
    # Both CrossedLMEResult and statsmodels MixedLMResults store conditional
    # residuals / fitted values in .resid / .fittedvalues.
    return np.asarray(model.resid, dtype=float), np.asarray(model.fittedvalues, dtype=float)


def hlm_resid(
    model,
    full_data: bool = True,
    type: str = "marginal",  # noqa: A002
    standardized: bool = False,
    level: int | str = 1,
) -> pd.DataFrame:
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

        if full_data:
            data = model.model.data.frame.reset_index(drop=True)
            return pd.concat([data, res_df], axis=1)

        return res_df

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
