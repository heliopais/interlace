"""Prototype + correctness check for Takahashi selected-inverse on pattern(L+L^T).

Run::

    uv run python docs/perf/probes/test_selected_inverse.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sksparse import cholmod  # type: ignore[import-untyped]


def selected_inv_takahashi(L_csc: sp.csc_matrix, perm: np.ndarray | None = None,
                            ) -> np.ndarray:
    """Dense q x q array containing A^{-1} on pattern(L + L^T), zero elsewhere.

    Implements the Erisman-Tinney / Takahashi recursion::

        Z[j,j] = 1/d_j^2 - 1/d_j * sum_{i>j, L[i,j]!=0} L[i,j] Z[i,j]
        Z[i,j] = -1/d_j * sum_{k>j, L[k,j]!=0} L[k,j] Z[k,i]   (i > j on pattern of L)

    *L_csc* is expected lower-triangular CSC (as returned by sksparse
    cholmod ``factor.L``) for the permuted matrix ``A[p][:,p] = L L^T``.
    When *perm* is provided, the returned Z is unpermuted to match A's
    original ordering.
    """
    L_csc = L_csc.tocsc(copy=False)
    L_csc.sort_indices()
    n = L_csc.shape[0]
    indptr = np.asarray(L_csc.indptr)
    indices = np.asarray(L_csc.indices)
    data = np.asarray(L_csc.data)
    Z = np.zeros((n, n))
    for j in range(n - 1, -1, -1):
        col_start = indptr[j]
        col_end = indptr[j + 1]
        row_idx = indices[col_start:col_end]
        col_data = data[col_start:col_end]
        # CSC of a lower-triangular matrix with sorted indices: first row is
        # the diagonal entry (since j is the smallest valid row index).
        if row_idx[0] != j:
            raise RuntimeError(
                f"L is not lower-triangular at column {j} (first row {row_idx[0]})"
            )
        d_j = col_data[0]
        d_inv = 1.0 / d_j
        R_j = row_idx[1:]
        v_j = col_data[1:]
        if R_j.size:
            block = Z[np.ix_(R_j, R_j)]
            z_col = -d_inv * (block @ v_j)
            Z[R_j, j] = z_col
            Z[j, R_j] = z_col
            Z[j, j] = d_inv * d_inv - d_inv * (v_j @ z_col)
        else:
            Z[j, j] = d_inv * d_inv
    if perm is not None:
        Z_orig = np.empty_like(Z)
        Z_orig[np.ix_(perm, perm)] = Z
        return Z_orig
    return Z


def _make_random_spd(q: int, density: float, seed: int = 0) -> sp.csc_matrix:
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, q, int(q * q * density / 2))
    cols = rng.integers(0, q, rows.size)
    data = rng.standard_normal(rows.size)
    M = sp.csc_matrix((data, (rows, cols)), shape=(q, q))
    A = (M @ M.T + sp.eye(q) * (q * 0.1)).tocsc()
    return A


def main():
    print("=== Correctness ===")
    for q, dens in [(20, 0.3), (80, 0.15), (180, 0.1)]:
        A = _make_random_spd(q, dens)
        f = cholmod.cho_factor(A)
        L = sp.csc_matrix(f.L)
        perm = np.asarray(f.perm)
        Z_takahashi = selected_inv_takahashi(L, perm=perm)
        Z_dense = np.linalg.inv(A.toarray())
        # Validate on pattern of (L+L^T) unpermuted.
        LLT_perm = (L + L.T).tocsc()
        # Unpermute pattern: in original ordering, pattern includes (perm[a], perm[b])
        # for each (a, b) in pattern(L+L^T).
        coo = LLT_perm.tocoo()
        i_orig = perm[coo.row]
        j_orig = perm[coo.col]
        diff = Z_takahashi[i_orig, j_orig] - Z_dense[i_orig, j_orig]
        print(f"q={q:>4d} dens={dens:.2f}  pattern_nnz={LLT_perm.nnz:>5d} "
              f"max abs diff on pattern = {np.max(np.abs(diff)):.3e}")

    print("\n=== Speed (q=180) vs full inverse ===")
    q, dens = 180, 0.1
    A = _make_random_spd(q, dens)
    f = cholmod.cho_factor(A)
    L = sp.csc_matrix(f.L)
    perm = np.asarray(f.perm)
    repeats = 7
    # Warmup
    selected_inv_takahashi(L, perm=perm)
    f.solve(np.eye(q), "A")

    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        selected_inv_takahashi(L, perm=perm)
        t.append(time.perf_counter() - t0)
    print(f"selected_inv_takahashi: median {np.median(t)*1e3:.3f} ms")

    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.asarray(f.solve(np.eye(q), "A"))
        t.append(time.perf_counter() - t0)
    print(f"cholmod.solve(I,'A'):    median {np.median(t)*1e3:.3f} ms")


if __name__ == "__main__":
    main()
