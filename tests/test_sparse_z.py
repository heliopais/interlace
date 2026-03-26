"""Tests for sparse_z.py — per-factor indicator matrices and joint Z."""

import numpy as np
import pytest
import scipy.sparse as sp

from interlace.sparse_z import build_indicator_matrix, build_joint_z


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
