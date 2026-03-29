"""Internal helpers for narwhals-based DataFrame boundary conversion.

All public interlace functions accept any narwhals-compatible frame
(pandas, polars, etc.) as input and return results in the same native type.
These helpers centralise that conversion logic.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd


def to_pandas(frame: Any) -> Any:
    """Convert any narwhals-compatible frame to a pandas DataFrame."""
    if isinstance(frame, pd.DataFrame):
        return frame
    nw_frame = nw.from_native(frame, eager_only=True)
    return pd.DataFrame({col: nw_frame[col].to_numpy() for col in nw_frame.columns})


def to_native(frame: Any, like: Any) -> Any:
    """Convert *frame* to the same native type as *like*.

    Works without polars when both *frame* and *like* are non-polars.
    """
    if isinstance(like, pd.DataFrame):
        # Target is pandas: ensure frame is pandas too
        if isinstance(frame, pd.DataFrame):
            return frame
        nw_df = nw.from_native(frame, eager_only=True)
        return pd.DataFrame({col: nw_df[col].to_numpy() for col in nw_df.columns})

    # Non-pandas target: convert via numpy column extraction
    native_ns = nw.get_native_namespace(like)
    nw_df = nw.from_native(frame, eager_only=True)
    col_dict = {col: nw_df[col].to_numpy() for col in nw_df.columns}
    return native_ns.DataFrame(col_dict)


def native_from_dict(col_dict: dict[str, Any], like: Any) -> Any:
    """Build a native DataFrame from a dict of arrays, matching the type of *like*."""
    if isinstance(like, pd.DataFrame):
        return pd.DataFrame(col_dict)
    native_ns = nw.get_native_namespace(like)
    return native_ns.DataFrame(col_dict)


def filter_rows(frame: Any, mask: np.ndarray) -> Any:
    """Filter rows of a native frame by a boolean numpy *mask*. Returns same type."""
    if isinstance(frame, pd.DataFrame):
        return frame[mask].reset_index(drop=True)
    # Non-pandas: assume polars-style .filter(boolean_series)
    native_ns = nw.get_native_namespace(frame)
    bool_col = native_ns.Series(mask)
    return frame.filter(bool_col)
