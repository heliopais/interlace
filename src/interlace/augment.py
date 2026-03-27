"""Combined residuals + influence diagnostics DataFrame."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from interlace.influence import hlm_influence
from interlace.residuals import hlm_resid


def hlm_augment(model: Any, level: int = 1, include_influence: bool = True) -> Any:
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
    Native DataFrame in the same type as the model's input data.
    """
    _ = level  # reserved for future multi-level support
    res_df = hlm_resid(model, type="conditional", full_data=True)

    if include_influence:
        infl_df = hlm_influence(model, level=1)
        # Both res_df and infl_df are in the native type; concat horizontally
        # via narwhals so the output type matches the input.
        nw_res = nw.from_native(res_df, eager_only=True)
        nw_infl = nw.from_native(infl_df, eager_only=True)
        combined = nw.concat([nw_res, nw_infl], how="horizontal")
        return nw.to_native(combined)

    return res_df
