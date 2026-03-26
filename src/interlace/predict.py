"""BLUP-based prediction for CrossedLMEResult.

In-sample:  result.predict() → result.fittedvalues  (X@beta + Z@b)
New data:   result.predict(newdata) → X_new@beta + sum of known BLUPs per factor
            Unknown group levels contribute 0 (shrink to the mean).
include_re: if False, return X_new@beta only (fixed-effects prediction).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import patsy

from interlace._frame import to_pandas as _to_pandas

if TYPE_CHECKING:
    from interlace.result import CrossedLMEResult


def predict(
    result: CrossedLMEResult,
    newdata: Any | None = None,
    include_re: bool = True,
) -> np.ndarray:
    """Return predictions from a fitted CrossedLMEResult.

    Parameters
    ----------
    result:
        A fitted ``CrossedLMEResult``.
    newdata:
        DataFrame to predict on.  If ``None``, returns ``result.fittedvalues``
        (in-sample conditional predictions).
    include_re:
        If ``True`` (default), add BLUP contributions for known group levels.
        If ``False``, return fixed-effects-only predictions (population mean).

    Returns
    -------
    np.ndarray of shape (n_obs,)
    """
    if newdata is None:
        return np.asarray(result.fittedvalues)

    newdata = _to_pandas(newdata)

    # Build fixed-effects design matrix for newdata using the same formula
    fe_formula = result.model.formula.split("~", 1)[1].strip()
    X_new = np.asarray(patsy.dmatrix(fe_formula, newdata, return_type="dataframe"))
    pred = X_new @ result.fe_params.values

    if not include_re:
        return np.asarray(pred)

    # Add BLUP contribution for each grouping factor
    group_cols = [result._gpgap_group_col] + list(result._gpgap_vc_cols)
    for col in group_cols:
        if col not in newdata.columns:
            continue
        blup_re = result.random_effects[col]
        if isinstance(blup_re, pd.DataFrame):
            # Random slopes: contribution = blup_intercept + sum(blup_slope_k * x_k)
            # re_df columns: ["(Intercept)", predictor1, predictor2, ...]
            predictors = list(blup_re.columns[1:])
            n_obs = len(newdata)
            contrib = np.zeros(n_obs)
            for i, level in enumerate(newdata[col]):
                if level not in blup_re.index:
                    continue  # unseen level → 0 (shrink to mean)
                blup_vec = blup_re.loc[level].to_numpy(dtype=float)
                z_row = np.array(
                    [1.0] + [float(newdata[p].iloc[i]) for p in predictors]
                )
                contrib[i] = blup_vec @ z_row
        else:
            # Intercept-only: map group level → scalar BLUP
            contrib = newdata[col].map(blup_re).fillna(0.0).to_numpy(dtype=float)
        pred = pred + contrib

    return np.asarray(pred)
