"""Combined residuals + influence diagnostics DataFrame."""

from __future__ import annotations

import pandas as pd

from interlace.influence import hlm_influence
from interlace.residuals import hlm_resid


def hlm_augment(model, level: int = 1, include_influence: bool = True) -> pd.DataFrame:
    """Combine residuals and (optionally) influence diagnostics into one DataFrame.

    Parameters
    ----------
    model:
        A ``CrossedLMEResult`` or statsmodels ``MixedLMResults`` object.
    level:
        Reserved for future multi-level support; currently only ``1`` is used.
    include_influence:
        If ``True`` (default), append Cook's D, MDFFITS, COVTRACE, COVRATIO,
        and RVC columns.  Set to ``False`` to skip the expensive refit loop.

    Returns
    -------
    pd.DataFrame
        Conditional residuals + original data + (optionally) influence stats.
    """
    _ = level  # reserved for future multi-level support
    res_df = hlm_resid(model, type="conditional", full_data=True)

    if include_influence:
        infl_df = hlm_influence(model, level=1)
        return pd.concat(
            [res_df.reset_index(drop=True), infl_df.reset_index(drop=True)], axis=1
        )

    return res_df
