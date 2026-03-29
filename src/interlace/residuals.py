"""Residual extraction for fitted linear mixed models.

Supports both ``CrossedLMEResult`` and statsmodels ``MixedLMResults``.

Note on statsmodels: ``MixedLMResults.fittedvalues`` includes predicted random
effects (conditional fitted values), so ``MixedLMResults.resid`` is the
conditional residual ``y - Xβ - Zu``.  Marginal residuals (``y - Xβ``) must be
computed explicitly from the stored endog and fixed-effects design matrix.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from interlace.result import CrossedLMEResult


def _is_crossed(model: Any) -> bool:
    return isinstance(model, CrossedLMEResult)


def _marginal(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(marginal_resid, marginal_fitted)`` = ``(y - Xβ, Xβ)``."""
    # Works for both CrossedLMEResult and statsmodels MixedLMResults since both
    # expose model.model.exog, model.model.endog, and model.fe_params.
    xbeta = np.asarray(model.model.exog @ np.asarray(model.fe_params), dtype=float)
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
    Native DataFrame in the same type as the model's input data.
    """
    native_frame = model.model.data.frame

    if level == 1:
        if type not in ("marginal", "conditional"):
            msg = "type must be 'marginal' or 'conditional'"
            raise ValueError(msg)

        res, fitted = _marginal(model) if type == "marginal" else _conditional(model)

        if standardized:
            res = res / np.sqrt(model.scale)

        res_dict: dict[str, Any] = {".resid": res, ".fitted": fitted}

        if full_data:
            # Choose the source frame: prefer the cached pandas frame (set when
            # pandas is installed); fall back to the native frame otherwise.
            _pandas_frame = getattr(model.model.data, "_pandas_frame", None)
            src = _pandas_frame if _pandas_frame is not None else native_frame
            nw_src = nw.from_native(src, eager_only=True)
            # Build column dict: original columns then residual columns
            combined: dict[str, Any] = {
                col: nw_src[col].to_numpy() for col in nw_src.columns
            }
            combined.update(res_dict)
            native_ns = nw.get_native_namespace(native_frame)
            return native_ns.DataFrame(combined)

        native_ns = nw.get_native_namespace(native_frame)
        return native_ns.DataFrame(res_dict)

    # --- group-level: return random effects ---
    if _is_crossed(model):
        if level not in model.random_effects:
            available = list(model.random_effects)
            msg = f"level '{level}' not found in random_effects; available: {available}"
            raise ValueError(msg)
        re_obj = model.random_effects[level]
        # re_obj: pd.Series (intercept-only) or pd.DataFrame (slopes)
        re_arr = np.asarray(re_obj.values if hasattr(re_obj, "values") else re_obj)
        re_index = (
            list(re_obj.index) if hasattr(re_obj, "index") else list(range(len(re_arr)))
        )

        level_col_name = str(level)
        result_dict: dict[str, Any] = {level_col_name: re_index}

        if re_arr.ndim == 1:
            # Intercept-only: single .ranef column named after the group factor
            re_name = re_obj.name if hasattr(re_obj, "name") else level
            result_dict[f".ranef.{re_name}"] = re_arr
        else:
            # Multi-term (slopes): one column per term
            if hasattr(re_obj, "columns"):
                term_names: list[str] = list(re_obj.columns)
            else:
                term_names = [str(i) for i in range(re_arr.shape[1])]
            for t_idx, term in enumerate(term_names):
                result_dict[f".ranef.{level}.{term}"] = re_arr[:, t_idx]

        native_ns = nw.get_native_namespace(native_frame)
        return native_ns.DataFrame(result_dict)

    # statsmodels: random_effects is {group_label: Series}
    import pandas as pd

    re = model.random_effects
    re_df = pd.DataFrame.from_dict(re, orient="index")
    re_df.index.name = level
    re_df.columns = [f".ranef.{col}" for col in re_df.columns]
    return re_df.reset_index()
