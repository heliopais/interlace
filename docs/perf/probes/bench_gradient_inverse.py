"""Micro-benchmark for interlace-mxzk Phase A.

Question: at q=180 (and across a sweep), is CHOLMOD's ``factor.solve(I, 'A')``
materially faster than the current ``splu(A11).solve(np.eye(q))`` used inside
``reml_gradient`` to build the dense ``A11_inv``?

If yes, Phase A (reuse the cached CHOLMOD factor) is worth pursuing. If not,
skip to Phase B (selected inverse on pattern(ZtZ)).

Run::

    uv run python docs/perf/probes/bench_gradient_inverse.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from interlace.profiled_reml import (  # noqa: E402
    _build_A11,
    _init_chol_factor,
    _precompute,
    _try_cholmod,
    make_lambda_diag,
    reml_gradient,
    reml_objective,
)
import scipy.linalg as la  # noqa: E402


def reml_gradient_phaseA(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    cache: dict,
) -> np.ndarray:
    """Prototype of reml_gradient that reuses the cached CHOLMOD factor for
    every solve, including the dense inverse against eye(q).

    Mirrors src/interlace/profiled_reml.py:reml_gradient line-for-line, with
    splu/_sparse_solve calls replaced by ``factor.solve(..., 'A')``.
    """
    ZtZ = sp.csc_matrix(cache["ZtZ"])
    ZtX = np.asarray(cache["ZtX"])
    Zty = np.asarray(cache["Zty"])
    XtX = np.asarray(cache["XtX"])
    Xty = np.asarray(cache["Xty"])
    yty = float(cache["yty"])

    factor = cache["chol_factor"]
    api = cache["chol_api"]

    lambda_diag = make_lambda_diag(theta, q_sizes)
    A11 = _build_A11(ZtZ, lambda_diag)
    # Refactor numerically (objective normally does this; here we do it once).
    if api == "new":
        factor.factorize(A11)
        solve_A = lambda b: np.asarray(factor.solve(b, "A"))  # noqa: E731
    else:
        factor.cholesky(A11)
        solve_A = lambda b: np.asarray(factor.solve_A(b))  # noqa: E731

    lZty = lambda_diag * Zty
    lZtX = lambda_diag[:, None] * ZtX
    n, p = X.shape
    q = A11.shape[0]

    c1 = solve_A(lZty)
    C_X = solve_A(lZtX)
    if c1.ndim == 2 and c1.shape[1] == 1:
        c1 = c1.ravel()
    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1
    beta_hat = la.solve(MX, rhs, assume_a="pos")
    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    MX_inv = np.linalg.inv(MX)

    # Dense inverse via CHOLMOD instead of splu.
    A11_inv = solve_A(np.eye(q))

    coo = ZtZ.tocoo()
    ZtZ_csr = ZtZ.tocsr()
    f = c1 - C_X @ beta_hat
    lf = lambda_diag * f
    Zt_resid = Zty - ZtX @ beta_hat

    grad = np.zeros(len(theta))
    q_start = 0
    for k, q_k in enumerate(q_sizes):
        q_end = q_start + q_k
        ek_row = (coo.row >= q_start) & (coo.row < q_end)
        ek_col = (coo.col >= q_start) & (coo.col < q_end)
        dA11_data = coo.data * (
            ek_row.astype(float) * lambda_diag[coo.col]
            + lambda_diag[coo.row] * ek_col.astype(float)
        )
        term1 = float(np.sum(A11_inv[coo.row, coo.col] * dA11_data))
        C_XT_dBk = C_X[q_start:q_end, :].T @ ZtX[q_start:q_end, :]
        dA11_sp = sp.csc_matrix((dA11_data, (coo.row, coo.col)), shape=(q, q))
        dA_CX = dA11_sp @ C_X
        C_XT_dA_CX = C_X.T @ dA_CX
        term2 = float(
            -2.0 * np.trace(MX_inv @ C_XT_dBk) + np.trace(MX_inv @ C_XT_dA_CX)
        )
        ZtZ_k_lf = np.asarray(ZtZ_csr[q_start:q_end, :] @ lf).ravel()
        d_yPy = float(2.0 * f[q_start:q_end] @ (ZtZ_k_lf - Zt_resid[q_start:q_end]))
        term3 = float((n - p) / yPy * d_yPy)
        grad[k] = term1 + term2 + term3
        q_start = q_end

    return grad


def _make_crossed_data(q1: int, q2: int, *, coverage: float = 1.0,
                        n_per_cell: int = 3, seed: int = 0):
    """Synthetic two-factor crossed-intercepts dataset.

    Parameters
    ----------
    coverage:
        Fraction of (g1, g2) cells that are observed.  ``1.0`` = full grid
        (dense off-diagonal ZtZ block); lower values produce realistic sparse
        patterns more like Sleepstudy / education data.

    Returns ``(y, X, Z, q_sizes)`` with ``q = q1 + q2``.
    """
    rng = np.random.default_rng(seed)
    n_cells = int(q1 * q2 * coverage)
    g1_cells = rng.integers(0, q1, size=n_cells)
    g2_cells = rng.integers(0, q2, size=n_cells)
    g1 = np.repeat(g1_cells, n_per_cell)
    g2 = np.repeat(g2_cells, n_per_cell)
    n = g1.size

    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    rows = np.arange(n)
    cols1 = g1
    cols2 = g2 + q1
    Z_data = np.ones(2 * n)
    Z_rows = np.concatenate([rows, rows])
    Z_cols = np.concatenate([cols1, cols2])
    Z = sp.csc_matrix((Z_data, (Z_rows, Z_cols)), shape=(n, q1 + q2))
    beta_true = np.array([1.0, 0.5])
    u_true = rng.standard_normal(q1 + q2) * 0.7
    y = X @ beta_true + Z @ u_true + rng.standard_normal(n) * 0.5
    return y, X, Z, [q1, q2]


def _build_A11_for_theta(y, X, Z, q_sizes, theta):
    cache = _precompute(y, X, Z)
    ZtZ = sp.csc_matrix(cache["ZtZ"])
    lambda_diag = make_lambda_diag(theta, q_sizes)
    return _build_A11(ZtZ, lambda_diag)


def _time(fn, *, repeats: int = 7, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples)


def bench_one(q1: int, q2: int, *, coverage: float = 1.0,
              repeats: int = 7) -> dict:
    q = q1 + q2
    y, X, Z, q_sizes = _make_crossed_data(q1, q2, coverage=coverage)
    theta = np.array([1.0, 0.5])  # arbitrary positive theta
    A11 = _build_A11_for_theta(y, X, Z, q_sizes, theta).tocsc()
    nnz_A11 = A11.nnz
    eye_q = np.eye(q)

    cholmod = _try_cholmod()
    factor, api = _init_chol_factor(cholmod, A11) if cholmod is not None else (None, None)

    # 1. Current: splu refactor + dense back-solve against I
    def _splu_full():
        lu = spla.splu(A11)
        return lu.solve(eye_q)

    t_splu = _time(_splu_full, repeats=repeats)

    # 2. Proposed Phase A.1: cached CHOLMOD factor + back-solve against I
    if factor is None:
        t_cholmod_full = float("nan")
        t_cholmod_refactor = float("nan")
    else:
        if api == "new":
            def _cholmod_refactor_only():
                factor.factorize(A11)
            def _cholmod_full():
                # *Reuse* the already-factored matrix (mirrors gradient seeing
                # the cached factor that the objective just produced).
                return np.asarray(factor.solve(eye_q, "A"))
        else:
            def _cholmod_refactor_only():
                factor.cholesky(A11)
            def _cholmod_full():
                return np.asarray(factor.solve_A(eye_q))

        # Make sure factor is up to date before timing the solve.
        _cholmod_refactor_only()
        t_cholmod_full = _time(_cholmod_full, repeats=repeats)
        t_cholmod_refactor = _time(_cholmod_refactor_only, repeats=repeats)

    # 3. Reference: cost of one obj-style sparse solve (lZty rhs is q-vector)
    rhs = np.random.default_rng(0).standard_normal(q)

    def _splu_one():
        lu = spla.splu(A11)
        return lu.solve(rhs)

    t_splu_one = _time(_splu_one, repeats=repeats)

    if factor is not None:
        if api == "new":
            def _cholmod_one():
                return np.asarray(factor.solve(rhs, "A"))
        else:
            def _cholmod_one():
                return np.asarray(factor.solve_A(rhs))
        t_cholmod_one = _time(_cholmod_one, repeats=repeats)
    else:
        t_cholmod_one = float("nan")

    # 4. End-to-end reml_gradient vs reml_objective wall-time (acceptance
    #    criterion: gradient <= objective at q=180).
    cache = _precompute(y, X, Z)
    # Prime cholmod factor in cache so reml_objective uses it.
    if factor is not None:
        cache["chol_factor"] = factor
        cache["chol_api"] = api

    def _obj():
        return reml_objective(theta, y, X, Z, q_sizes, cache)

    def _grad():
        return reml_gradient(theta, y, X, Z, q_sizes, cache)

    def _grad_phaseA():
        return reml_gradient_phaseA(theta, y, X, Z, q_sizes, cache)

    t_obj = _time(_obj, repeats=repeats)
    t_grad = _time(_grad, repeats=repeats)
    if factor is not None:
        # Sanity: phase-A grad must match current grad numerically.
        g_curr = reml_gradient(theta, y, X, Z, q_sizes, cache)
        g_phaseA = reml_gradient_phaseA(theta, y, X, Z, q_sizes, cache)
        if not np.allclose(g_curr, g_phaseA, rtol=1e-8, atol=1e-10):
            raise RuntimeError(
                f"Phase-A gradient mismatch at q={q}, cov={coverage}: "
                f"max abs diff = {np.max(np.abs(g_curr - g_phaseA))}"
            )
        t_grad_phaseA = _time(_grad_phaseA, repeats=repeats)
    else:
        t_grad_phaseA = float("nan")

    return dict(
        q=q,
        q1=q1,
        q2=q2,
        coverage=coverage,
        nnz_A11=nnz_A11,
        density=nnz_A11 / (q * q),
        api=api,
        t_obj_ms=t_obj * 1e3,
        t_grad_ms=t_grad * 1e3,
        t_grad_phaseA_ms=t_grad_phaseA * 1e3,
        grad_over_obj=t_grad / t_obj if t_obj > 0 else float("nan"),
        grad_phaseA_over_obj=t_grad_phaseA / t_obj if t_obj > 0 else float("nan"),
        grad_speedup=t_grad / t_grad_phaseA if t_grad_phaseA > 0 else float("nan"),
        t_splu_full_ms=t_splu * 1e3,
        t_cholmod_refactor_ms=t_cholmod_refactor * 1e3,
        t_cholmod_full_ms=t_cholmod_full * 1e3,
        t_cholmod_full_total_ms=(t_cholmod_refactor + t_cholmod_full) * 1e3
        if not np.isnan(t_cholmod_full)
        else float("nan"),
        t_splu_one_ms=t_splu_one * 1e3,
        t_cholmod_one_ms=t_cholmod_one * 1e3,
        speedup_full=(t_splu / t_cholmod_full) if t_cholmod_full > 0 else float("nan"),
        speedup_full_with_refactor=(
            t_splu / (t_cholmod_full + t_cholmod_refactor)
            if t_cholmod_full > 0
            else float("nan")
        ),
    )


def main():
    # Two coverage regimes per q:
    #   coverage=1.0  -> dense off-diagonal block (worst case for any sparse method)
    #   coverage=0.2  -> realistic crossed-data sparsity
    sweeps = [
        (40, 40, 1.0),
        (40, 40, 0.2),
        (90, 90, 1.0),
        (90, 90, 0.2),
        (150, 150, 1.0),
        (150, 150, 0.2),
    ]
    print(
        f"{'q':>5} {'cov':>5} {'dens':>6} "
        f"{'obj':>9} {'grad':>9} {'gradA':>9} "
        f"{'g/o':>6} {'gA/o':>6} {'g/gA':>6}"
    )
    print("-" * 80)
    rows: list[dict] = []
    for q1, q2, cov in sweeps:
        r = bench_one(q1, q2, coverage=cov)
        rows.append(r)
        print(
            f"{r['q']:>5d} {r['coverage']:>5.2f} "
            f"{r['density']*100:>5.1f}% "
            f"{r['t_obj_ms']:>8.2f}ms "
            f"{r['t_grad_ms']:>8.2f}ms "
            f"{r['t_grad_phaseA_ms']:>8.2f}ms "
            f"{r['grad_over_obj']:>5.2f}x "
            f"{r['grad_phaseA_over_obj']:>5.2f}x "
            f"{r['grad_speedup']:>5.2f}x"
        )

    print()
    print("Legend:")
    print("  obj    = end-to-end reml_objective call (cached cholmod)")
    print("  grad   = current reml_gradient (splu refactor + dense inverse)")
    print("  gradA  = Phase-A reml_gradient prototype (cached cholmod for all solves)")
    print("  g/o    = current grad / obj           (current ratio)")
    print("  gA/o   = phase-A grad / obj           (target: <= 1.0)")
    print("  g/gA   = speedup phase-A delivers     (current grad / phase-A grad)")


if __name__ == "__main__":
    main()
