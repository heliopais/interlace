"""Internal helpers for narwhals-based DataFrame boundary conversion.

All public interlace functions accept any narwhals-compatible frame
(pandas, polars, etc.) as input and return results in the same native type.
These two helpers centralise that conversion logic.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import pandas as pd


def to_pandas(frame: Any) -> pd.DataFrame:
    """Convert any narwhals-compatible frame to a pandas DataFrame.

    Extracts columns via narwhals ``to_numpy()`` rather than going through
    ``to_pandas()`` so that pyarrow is not required for string columns.
    """
    if isinstance(frame, pd.DataFrame):
        return frame
    nw_frame = nw.from_native(frame, eager_only=True)
    return pd.DataFrame(
        {col: nw_frame[col].to_numpy() for col in nw_frame.columns}
    )


def to_native(df_pandas: pd.DataFrame, like: Any) -> Any:
    """Convert a pandas DataFrame to the native type of *like*.

    If *like* is already a pandas DataFrame the input is returned unchanged.
    Constructs the target frame column-by-column from numpy arrays so that
    pyarrow is not required.
    """
    native_ns = nw.get_native_namespace(like)
    if native_ns is pd:
        return df_pandas
    # Build target-library DataFrame from numpy arrays (no pyarrow needed).
    nw_df = nw.from_native(df_pandas, eager_only=True)
    col_dict = {col: nw_df[col].to_numpy() for col in nw_df.columns}
    return native_ns.DataFrame(col_dict)
