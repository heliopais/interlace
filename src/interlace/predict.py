"""BLUP-based prediction for CrossedLMEResult.

In-sample:  result.predict() → result.fittedvalues  (X@beta + Z@b)
New data:   result.predict(newdata) → X_new@beta + sum of known BLUPs per factor
            Unknown group levels contribute 0 (shrink to the mean).
include_re: if False, return X_new@beta only (fixed-effects prediction).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import formulaic
import narwhals as nw
import numpy as np

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
        (in-sample conditional predictions).  Any narwhals-compatible frame
        (pandas, polars, …) is accepted.
    include_re:
        If ``True`` (default), add BLUP contributions for known group levels.
        If ``False``, return fixed-effects-only predictions (population mean).

    Returns
    -------
    np.ndarray of shape (n_obs,)
    """
    if newdata is None:
        return np.asarray(result.fittedvalues)

    nw_new = nw.from_native(newdata, eager_only=True)

    # Build fixed-effects design matrix using formulaic (same as fitting path).
    fe_formula = result.model.formula.split("~", 1)[1].strip()
    X_new_df = formulaic.model_matrix(fe_formula, nw_new)
    # Reindex to fitting column order before dot product (column order can differ).
    X_new = np.asarray(X_new_df[list(result.fe_params.index)])
    pred = X_new @ np.asarray(result.fe_params)

    if not include_re:
        return np.asarray(pred)

    # Add BLUP contribution for each grouping factor
    group_cols = [result._gpgap_group_col] + list(result._gpgap_vc_cols)
    for col in group_cols:
        if col not in nw_new.columns:
            continue
        blup_re = result.random_effects[col]
        col_vals = nw_new[col].to_numpy()

        # Detect whether blup_re is a pandas DataFrame (random slopes, pandas path)
        try:
            import pandas as pd

            if isinstance(blup_re, pd.DataFrame):
                # Random slopes: contribution = blup_intercept + sum(blup_slope_k * x_k)
                predictors = list(blup_re.columns[1:])
                n_obs = len(col_vals)
                contrib = np.zeros(n_obs)
                for i, level in enumerate(col_vals):
                    if level not in blup_re.index:
                        continue  # unseen level → 0
                    blup_vec = blup_re.loc[level].to_numpy(dtype=float)
                    z_row = np.array(
                        [1.0] + [float(nw_new[p].to_numpy()[i]) for p in predictors]
                    )
                    contrib[i] = blup_vec @ z_row
                pred = pred + contrib
                continue
            if isinstance(blup_re, pd.Series):
                # Intercept-only, pandas path: map via dict lookup
                lookup = blup_re.to_dict()
                contrib = np.array([lookup.get(v, 0.0) for v in col_vals], dtype=float)
                pred = pred + contrib
                continue
        except ImportError:
            pass

        # Pandas-free path: blup_re is _SimpleRE or numpy array
        if hasattr(blup_re, "values") and hasattr(blup_re, "index"):
            # _SimpleRE
            lookup = dict(zip(blup_re.index, blup_re.values.tolist(), strict=True))
        else:
            # raw numpy array — no index info available, skip
            continue
        contrib = np.array([lookup.get(v, 0.0) for v in col_vals], dtype=float)
        pred = pred + contrib

    return np.asarray(pred)
