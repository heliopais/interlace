"""Tests for sparse_z.py — per-factor indicator matrices and joint Z."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from interlace.formula import RandomEffectSpec
from interlace.sparse_z import (
    build_indicator_matrix,
    build_joint_z,
    build_joint_z_from_specs,
    build_z_block,
)


@pytest.fixture()
def three_group_codes() -> tuple[np.ndarray, int]:
    # 12 observations, 3 groups: [0,1,2,0,1,2,...]
    codes = np.tile(np.arange(3), 4).astype(np.intp)
    return codes, 3


@pytest.fixture()
def two_group_codes() -> tuple[np.ndarray, int]:
    codes = np.tile(np.arange(2), 6).astype(np.intp)
    return codes, 2


class TestBuildIndicatorMatrix:
    def test_returns_csc_matrix(
        self, three_group_codes: tuple[np.ndarray, int]
    ) -> None:
        codes, n_levels = three_group_codes
        Z = build_indicator_matrix(codes, n_levels)
        assert sp.issparse(Z)
        assert isinstance(Z, sp.csc_matrix)

    def test_shape(self, three_group_codes: tuple[np.ndarray, int]) -> None:
        codes, n_levels = three_group_codes
        Z = build_indicator_matrix(codes, n_levels)
        assert Z.shape == (12, 3)

    def test_binary_entries(self, three_group_codes: tuple[np.ndarray, int]) -> None:
        codes, n_levels = three_group_codes
        Z = build_indicator_matrix(codes, n_levels)
        data = Z.data
        assert set(data).issubset({1.0})

    def test_exactly_one_per_row(
        self, three_group_codes: tuple[np.ndarray, int]
    ) -> None:
        codes, n_levels = three_group_codes
        Z = build_indicator_matrix(codes, n_levels)
        row_sums = np.asarray(Z.sum(axis=1)).squeeze()
        np.testing.assert_array_equal(row_sums, np.ones(12))

    def test_correct_column_assignment(
        self, three_group_codes: tuple[np.ndarray, int]
    ) -> None:
        codes, n_levels = three_group_codes
        Z = build_indicator_matrix(codes, n_levels)
        dense = Z.toarray()
        for i, code in enumerate(codes):
            assert dense[i, code] == 1.0
            assert dense[i, :].sum() == 1.0

    def test_column_sums(self, three_group_codes: tuple[np.ndarray, int]) -> None:
        codes, n_levels = three_group_codes
        Z = build_indicator_matrix(codes, n_levels)
        col_sums = np.asarray(Z.sum(axis=0)).squeeze()
        np.testing.assert_array_equal(col_sums, [4.0, 4.0, 4.0])


class TestBuildJointZ:
    def test_shape(
        self,
        three_group_codes: tuple[np.ndarray, int],
        two_group_codes: tuple[np.ndarray, int],
    ) -> None:
        factors = [
            ("g1", three_group_codes[0], three_group_codes[1]),
            ("g2", two_group_codes[0], two_group_codes[1]),
        ]
        Z = build_joint_z(factors)
        assert Z.shape == (12, 5)  # 3 + 2 columns

    def test_returns_csc_matrix(
        self, three_group_codes: tuple[np.ndarray, int]
    ) -> None:
        factors = [("g1", three_group_codes[0], three_group_codes[1])]
        Z = build_joint_z(factors)
        assert isinstance(Z, sp.csc_matrix)

    def test_single_factor_identical_to_indicator(
        self, three_group_codes: tuple[np.ndarray, int]
    ) -> None:
        codes, n_levels = three_group_codes
        Z_joint = build_joint_z([("g1", codes, n_levels)])
        Z_single = build_indicator_matrix(codes, n_levels)
        np.testing.assert_array_equal(Z_joint.toarray(), Z_single.toarray())

    def test_block_structure(
        self,
        three_group_codes: tuple[np.ndarray, int],
        two_group_codes: tuple[np.ndarray, int],
    ) -> None:
        factors = [
            ("g1", three_group_codes[0], three_group_codes[1]),
            ("g2", two_group_codes[0], two_group_codes[1]),
        ]
        Z = build_joint_z(factors)
        dense = Z.toarray()
        # Left block (cols 0-2) matches g1 indicator
        Z1 = build_indicator_matrix(three_group_codes[0], three_group_codes[1])
        np.testing.assert_array_equal(dense[:, :3], Z1.toarray())
        # Right block (cols 3-4) matches g2 indicator
        Z2 = build_indicator_matrix(two_group_codes[0], two_group_codes[1])
        np.testing.assert_array_equal(dense[:, 3:], Z2.toarray())


# ---------------------------------------------------------------------------
# Fixtures shared by the new tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def slope_df() -> tuple[pd.DataFrame, np.ndarray, int]:
    """12 obs, 3 groups, one predictor x. Returns (df, codes, n_levels)."""
    rng = np.random.default_rng(7)
    codes = np.tile(np.arange(3), 4).astype(np.intp)
    groups = np.where(codes == 0, "a", np.where(codes == 1, "b", "c"))
    x = rng.normal(size=12)
    df = pd.DataFrame({"g": groups, "x": x})
    return df, codes, 3


# ---------------------------------------------------------------------------
# build_z_block
# ---------------------------------------------------------------------------


class TestBuildZBlock:
    def test_intercept_only_matches_indicator(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        spec = RandomEffectSpec(
            group="g", predictors=[], intercept=True, correlated=True
        )
        Z = build_z_block(spec, df, codes, n_levels)
        Z_expected = build_indicator_matrix(codes, n_levels)
        np.testing.assert_array_equal(Z.toarray(), Z_expected.toarray())

    def test_intercept_and_slope_shape(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=True, correlated=True
        )
        Z = build_z_block(spec, df, codes, n_levels)
        assert Z.shape == (12, 6)  # 3 intercept cols + 3 slope cols

    def test_intercept_columns_correct(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=True, correlated=True
        )
        Z = build_z_block(spec, df, codes, n_levels)
        dense = Z.toarray()
        Z_ind = build_indicator_matrix(codes, n_levels).toarray()
        np.testing.assert_array_equal(dense[:, :3], Z_ind)

    def test_slope_column_values(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=True, correlated=True
        )
        Z = build_z_block(spec, df, codes, n_levels)
        dense = Z.toarray()
        x = df["x"].values
        # Slope cols: Z[i, 3 + codes[i]] == x[i], all other slope cols == 0
        for i in range(12):
            assert dense[i, 3 + codes[i]] == pytest.approx(x[i])
            for k in range(3):
                if k != codes[i]:
                    assert dense[i, 3 + k] == 0.0

    def test_slope_only_shape(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=False, correlated=True
        )
        Z = build_z_block(spec, df, codes, n_levels)
        assert Z.shape == (12, 3)

    def test_slope_only_values(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=False, correlated=True
        )
        Z = build_z_block(spec, df, codes, n_levels)
        dense = Z.toarray()
        x = df["x"].values
        for i in range(12):
            assert dense[i, codes[i]] == pytest.approx(x[i])

    def test_two_predictors_shape(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        df = df.copy()
        df["x2"] = np.arange(12, dtype=float)
        spec = RandomEffectSpec(
            group="g", predictors=["x", "x2"], intercept=True, correlated=True
        )
        Z = build_z_block(spec, df, codes, n_levels)
        assert Z.shape == (12, 9)  # 3 terms × 3 levels

    def test_returns_csc_matrix(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=True, correlated=True
        )
        Z = build_z_block(spec, df, codes, n_levels)
        assert isinstance(Z, sp.csc_matrix)


# ---------------------------------------------------------------------------
# build_joint_z_from_specs
# ---------------------------------------------------------------------------


class TestBuildJointZFromSpecs:
    def test_single_intercept_only_matches_build_joint_z(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, codes, n_levels = slope_df
        spec = RandomEffectSpec(
            group="g", predictors=[], intercept=True, correlated=True
        )
        Z_new = build_joint_z_from_specs([spec], df)
        Z_old = build_joint_z([("g", codes, n_levels)])
        np.testing.assert_array_equal(Z_new.toarray(), Z_old.toarray())

    def test_two_intercept_only_specs_match_build_joint_z(self) -> None:
        rng = np.random.default_rng(1)
        n = 12
        g1 = np.tile(["a", "b", "c"], 4)
        g2 = np.tile(["x", "y"], 6)
        df = pd.DataFrame({"g1": g1, "g2": g2, "y": rng.normal(size=n)})
        codes1 = pd.factorize(df["g1"], sort=True)[0].astype(np.intp)
        codes2 = pd.factorize(df["g2"], sort=True)[0].astype(np.intp)

        specs = [
            RandomEffectSpec(
                group="g1", predictors=[], intercept=True, correlated=True
            ),
            RandomEffectSpec(
                group="g2", predictors=[], intercept=True, correlated=True
            ),
        ]
        Z_new = build_joint_z_from_specs(specs, df)
        Z_old = build_joint_z([("g1", codes1, 3), ("g2", codes2, 2)])
        np.testing.assert_array_equal(Z_new.toarray(), Z_old.toarray())

    def test_mixed_specs_column_count(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, _, _ = slope_df
        df = df.copy()
        df["g2"] = np.tile(["x", "y"], 6)
        specs = [
            RandomEffectSpec(
                group="g", predictors=["x"], intercept=True, correlated=True
            ),
            RandomEffectSpec(
                group="g2", predictors=[], intercept=True, correlated=True
            ),
        ]
        Z = build_joint_z_from_specs(specs, df)
        # g has 3 levels × 2 terms = 6 cols; g2 has 2 levels × 1 term = 2 cols
        assert Z.shape == (12, 8)

    def test_returns_csc_matrix(
        self, slope_df: tuple[pd.DataFrame, np.ndarray, int]
    ) -> None:
        df, _, _ = slope_df
        specs = [
            RandomEffectSpec(group="g", predictors=[], intercept=True, correlated=True)
        ]
        Z = build_joint_z_from_specs(specs, df)
        assert isinstance(Z, sp.csc_matrix)
