"""Tests for the in-place CSC scaling helpers in glmm_laplace.

These replace ``Z @ sp.diags(d)`` and ``sp.diags(d) @ M`` in the PIRLS
inner loop without going through scipy's sparse-matmul machinery.
"""

import numpy as np
import scipy.sparse as sp

from interlace.glmm_laplace import _scale_columns_csc, _scale_rows_csc


def _rand_csc(
    rng: np.random.Generator, shape: tuple[int, int], density: float = 0.4
) -> sp.csc_matrix:
    n, m = shape
    dense = rng.standard_normal(size=(n, m))
    dense[rng.uniform(size=(n, m)) > density] = 0.0
    return sp.csc_matrix(dense)


class TestScaleColumnsCSC:
    def test_matches_diag_matmul_dense(self) -> None:
        rng = np.random.default_rng(7)
        Z = _rand_csc(rng, (12, 5))
        col_factors = rng.uniform(0.1, 2.0, size=5)
        out = _scale_columns_csc(Z, col_factors)
        expected = Z.toarray() * col_factors[None, :]
        np.testing.assert_allclose(out.toarray(), expected)

    def test_matches_sparse_diag_matmul(self) -> None:
        rng = np.random.default_rng(8)
        Z = _rand_csc(rng, (20, 7))
        col_factors = rng.uniform(0.1, 2.0, size=7)
        out = _scale_columns_csc(Z, col_factors)
        expected = (Z @ sp.diags(col_factors, format="csc")).toarray()
        np.testing.assert_allclose(out.toarray(), expected)

    def test_does_not_mutate_input(self) -> None:
        rng = np.random.default_rng(9)
        Z = _rand_csc(rng, (8, 4))
        Z_data_before = Z.data.copy()
        _scale_columns_csc(Z, np.array([2.0, 3.0, 5.0, 7.0]))
        np.testing.assert_array_equal(Z.data, Z_data_before)

    def test_preserves_sparsity_pattern(self) -> None:
        rng = np.random.default_rng(10)
        Z = _rand_csc(rng, (10, 5))
        out = _scale_columns_csc(Z, np.ones(5))
        np.testing.assert_array_equal(out.indices, Z.indices)
        np.testing.assert_array_equal(out.indptr, Z.indptr)


class TestScaleRowsCSC:
    def test_matches_diag_matmul_dense(self) -> None:
        rng = np.random.default_rng(11)
        M = _rand_csc(rng, (15, 6))
        row_factors = rng.uniform(0.1, 2.0, size=15)
        out = _scale_rows_csc(M, row_factors)
        expected = M.toarray() * row_factors[:, None]
        np.testing.assert_allclose(out.toarray(), expected)

    def test_matches_sparse_diag_matmul(self) -> None:
        rng = np.random.default_rng(12)
        M = _rand_csc(rng, (20, 8))
        row_factors = rng.uniform(0.1, 2.0, size=20)
        out = _scale_rows_csc(M, row_factors)
        expected = (sp.diags(row_factors, format="csc") @ M).toarray()
        np.testing.assert_allclose(out.toarray(), expected)

    def test_does_not_mutate_input(self) -> None:
        rng = np.random.default_rng(13)
        M = _rand_csc(rng, (10, 4))
        M_data_before = M.data.copy()
        _scale_rows_csc(M, np.linspace(1.0, 3.0, 10))
        np.testing.assert_array_equal(M.data, M_data_before)

    def test_preserves_sparsity_pattern(self) -> None:
        rng = np.random.default_rng(14)
        M = _rand_csc(rng, (12, 6))
        out = _scale_rows_csc(M, np.ones(12))
        np.testing.assert_array_equal(out.indices, M.indices)
        np.testing.assert_array_equal(out.indptr, M.indptr)
