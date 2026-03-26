"""Combined residuals + influence diagnostics DataFrame."""

from __future__ import annotations

from typing import Any

import pandas as pd

from interlace._frame import to_native as _to_native
from interlace._frame import to_pandas as _to_pandas
from interlace.influence import hlm_influence
from interlace.residuals import hlm_resid


def hlm_augment(
    model: Any, level: int = 1, include_influence: bool = True
) -> pd.DataFrame:
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
    native_frame = model.model.data.frame
    res_df = hlm_resid(model, type="conditional", full_data=True)

    if include_influence:
        infl_df = hlm_influence(model, level=1)
        # Normalise both to pandas for concat, then convert back to native type.
        pd_res = _to_pandas(res_df)
        combined = pd.concat(
            [pd_res.reset_index(drop=True), infl_df.reset_index(drop=True)], axis=1
        )
        return _to_native(combined, like=native_frame)

    return res_df
