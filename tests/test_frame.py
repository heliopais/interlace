"""Unit tests for interlace._frame internal helpers.

Covers: to_pandas, to_native, native_from_dict, filter_rows
for both pandas and polars inputs, including the ImportError fallback paths.
"""

from __future__ import annotations

import unittest.mock
from typing import Any

import numpy as np
import pandas as pd
import pytest

polars = pytest.importorskip("polars")

from interlace._frame import filter_rows, native_from_dict, to_native, to_pandas  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pd_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


@pytest.fixture
def pl_df(pd_df) -> Any:
    return polars.from_pandas(pd_df)


@pytest.fixture
def mask() -> np.ndarray:
    return np.array([True, False, True])


# ---------------------------------------------------------------------------
# to_pandas
# ---------------------------------------------------------------------------


class TestToPandas:
    def test_pandas_passthrough(self, pd_df):
        result = to_pandas(pd_df)
        assert result is pd_df

    def test_polars_converts_to_pandas(self, pl_df, pd_df):
        result = to_pandas(pl_df)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, pd_df)

    def test_raises_importerror_when_pandas_missing(self, pl_df):
        with unittest.mock.patch.dict("sys.modules", {"pandas": None}):
            with pytest.raises(ImportError, match="pandas is required"):
                to_pandas(pl_df)


# ---------------------------------------------------------------------------
# to_native
# ---------------------------------------------------------------------------


class TestToNative:
    def test_pandas_like_polars_frame_returns_pandas(self, pl_df, pd_df):
        result = to_native(pl_df, like=pd_df)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, pd_df)

    def test_pandas_like_pandas_frame_passthrough(self, pd_df):
        result = to_native(pd_df, like=pd_df)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, pd_df)

    def test_polars_like_polars_frame_returns_polars(self, pl_df):
        result = to_native(pl_df, like=pl_df)
        assert isinstance(result, polars.DataFrame)
        assert result.equals(pl_df)

    def test_no_pandas_polars_to_polars(self, pl_df):
        """When pandas is not installed, falls through to polars path."""
        with unittest.mock.patch.dict("sys.modules", {"pandas": None}):
            result = to_native(pl_df, like=pl_df)
        assert isinstance(result, polars.DataFrame)
        assert result.equals(pl_df)


# ---------------------------------------------------------------------------
# native_from_dict
# ---------------------------------------------------------------------------


class TestNativeFromDict:
    def test_pandas_like_returns_pandas(self, pd_df):
        col_dict = {"a": np.array([1, 2, 3]), "b": np.array([4.0, 5.0, 6.0])}
        result = native_from_dict(col_dict, like=pd_df)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, pd_df)

    def test_polars_like_returns_polars(self, pl_df, pd_df):
        col_dict = {"a": np.array([1, 2, 3]), "b": np.array([4.0, 5.0, 6.0])}
        result = native_from_dict(col_dict, like=pl_df)
        assert isinstance(result, polars.DataFrame)
        assert result.equals(pl_df)

    def test_no_pandas_polars_like_returns_polars(self, pl_df):
        col_dict = {"a": np.array([1, 2, 3]), "b": np.array([4.0, 5.0, 6.0])}
        with unittest.mock.patch.dict("sys.modules", {"pandas": None}):
            result = native_from_dict(col_dict, like=pl_df)
        assert isinstance(result, polars.DataFrame)
        assert result.equals(pl_df)


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


class TestFilterRows:
    def test_pandas_filters_correctly(self, pd_df, mask):
        result = filter_rows(pd_df, mask)
        assert isinstance(result, pd.DataFrame)
        assert list(result["a"]) == [1, 3]
        assert result.index.tolist() == [0, 1]  # reset_index applied

    def test_polars_filters_correctly(self, pl_df, mask):
        result = filter_rows(pl_df, mask)
        assert isinstance(result, polars.DataFrame)
        assert result["a"].to_list() == [1, 3]

    def test_no_pandas_polars_filters_correctly(self, pl_df, mask):
        with unittest.mock.patch.dict("sys.modules", {"pandas": None}):
            result = filter_rows(pl_df, mask)
        assert isinstance(result, polars.DataFrame)
        assert result["a"].to_list() == [1, 3]

    def test_all_true_mask(self, pd_df, pl_df):
        mask = np.ones(3, dtype=bool)
        assert len(filter_rows(pd_df, mask)) == 3
        assert len(filter_rows(pl_df, mask)) == 3

    def test_all_false_mask(self, pd_df, pl_df):
        mask = np.zeros(3, dtype=bool)
        assert len(filter_rows(pd_df, mask)) == 0
        assert len(filter_rows(pl_df, mask)) == 0
