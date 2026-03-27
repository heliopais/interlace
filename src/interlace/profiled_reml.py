"""Profiled REML estimation for linear mixed models with crossed random effects.

Implements the Bates et al. (2015) profiled REML criterion using:
- Lambda_theta parameterisation (diagonal for intercept-only; Cholesky Kronecker
  product or block-diagonal for random slopes)
- Sparse Cholesky factorisation (scipy.sparse.linalg.splu)
- L-BFGS-B optimisation over theta

References
----------
Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015).
Fitting Linear Mixed-Effects Models Using lme4.
Journal of Statistical Software, 67(1), 1-48.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    from interlace.formula import RandomEffectSpec

# ---------------------------------------------------------------------------
# Public dataclass for fit results
# ---------------------------------------------------------------------------


@dataclass
class REMLResult:
    beta: np.ndarray
    theta: np.ndarray
    sigma2: float
    converged: bool
    llf: float
    aic: float
    bic: float
    nobs: int
    nparams: int  # p (FE) + k (RE variances) + 1 (sigma²)
    specs: list[RandomEffectSpec] | None = None
    n_levels: list[int] | None = None


# ---------------------------------------------------------------------------
# Lambda_theta parameterisation
# ---------------------------------------------------------------------------


def n_theta_for_spec(n_terms: int, correlated: bool) -> int:
    """Number of theta parameters for a single random effect spec.

    Parameters
    ----------
    n_terms:
        Total number of random effect terms per group level (intercept + slopes).
    correlated:
        True → full lower-triangular Cholesky; False → diagonal (independent).

    Returns
    -------
    int
        ``1`` when n_terms == 1 (regardless of correlated flag);
        ``n_terms * (n_terms + 1) // 2`` for correlated multi-term;
        ``n_terms`` for independent multi-term.
    """
    if n_terms == 1:
        return 1
    return n_terms * (n_terms + 1) // 2 if correlated else n_terms


def make_lambda(
    theta: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
) -> sp.csc_matrix:
    """Build the block-diagonal Lambda_theta sparse matrix.

    For each spec j with ``p_j = spec.n_terms`` terms and ``q_j`` group levels:

    * ``p_j == 1``: ``Lambda_j = theta_j * I_{q_j}`` (scalar, unchanged behaviour)
    * ``p_j > 1``, correlated: ``Lambda_j = L_j ⊗ I_{q_j}`` where ``L_j`` is a
      ``p_j × p_j`` lower-triangular matrix whose entries are the
      ``p_j*(p_j+1)/2`` theta parameters in row-major lower-tri order.
    * ``p_j > 1``, independent (``||``): ``Lambda_j = blkdiag(theta_j[0] * I_{q_j},
      theta_j[1] * I_{q_j}, ...)`` — one scalar per term.

    Column ordering within each Z block is assumed to be term-first (all ``q_j``
    intercept columns, then all ``q_j`` columns per slope predictor), matching
    :func:`interlace.sparse_z.build_z_block`.

    Parameters
    ----------
    theta:
        Flat array of all variance parameters (concatenated across specs).
    specs:
        Random effect specifications in the same order as the Z blocks.
    n_levels:
        Number of group levels for each spec (``q_j``).

    Returns
    -------
    scipy.sparse.csc_matrix of shape
    ``(sum(p_j * q_j), sum(p_j * q_j))``.
    """
    blocks: list[sp.csc_matrix] = []
    theta_idx = 0
    for spec, q_j in zip(specs, n_levels, strict=True):
        p_j = spec.n_terms
        n_theta_j = n_theta_for_spec(p_j, spec.correlated)
        theta_j = theta[theta_idx : theta_idx + n_theta_j]
        theta_idx += n_theta_j

        if p_j == 1:
            block: sp.csc_matrix = (theta_j[0] * sp.eye(q_j, format="csc")).tocsc()
        elif spec.correlated:
            # Build lower-triangular L_j from theta_j (row-major lower-tri order)
            L_j = np.zeros((p_j, p_j))
            idx = 0
            for row in range(p_j):
                for col in range(row + 1):
                    L_j[row, col] = theta_j[idx]
                    idx += 1
            block = sp.kron(sp.csc_matrix(L_j), sp.eye(q_j, format="csc"), format="csc")
        else:
            # Independent: blkdiag(theta_j[k] * I_{q_j}) for each term k
            sub_blocks = [
                (theta_j[k] * sp.eye(q_j, format="csc")).tocsc() for k in range(p_j)
            ]
            block = sp.block_diag(sub_blocks, format="csc")

        blocks.append(block)

    return sp.block_diag(blocks, format="csc")


def make_lambda_diag(theta: np.ndarray, q_sizes: list[int]) -> np.ndarray:
    """Build the diagonal of the Lambda_theta block-diagonal matrix.

    For crossed random intercepts, Lambda_theta is block-diagonal with
    blocks ``theta[j] * I_{q_j}``.  Its diagonal is therefore ``theta[j]``
    repeated ``q_sizes[j]`` times for each factor j.

    Parameters
    ----------
    theta:
        Relative covariance parameters, one per grouping factor.
    q_sizes:
        Number of levels for each grouping factor.

    Returns
    -------
    np.ndarray of length ``sum(q_sizes)``.
    """
    return np.repeat(theta, q_sizes)


# ---------------------------------------------------------------------------
# CHOLMOD optional import
# ---------------------------------------------------------------------------


def _try_cholmod() -> Any:
    """Return the ``sksparse.cholmod`` module, or ``None`` if not installed."""
    try:
        from sksparse import cholmod  # type: ignore[import-not-found]

        return cholmod
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Sparse Cholesky helpers
# ---------------------------------------------------------------------------


def sparse_chol_logdet(M: sp.csc_matrix) -> float:
    """Compute log|M| for a sparse symmetric positive-definite matrix M.

    Uses ``scipy.sparse.linalg.splu`` (SuperLU).  For a PD matrix the LU
    factorisation satisfies ``PA = LU`` with L unit-lower-triangular, so
    ``log|M| = sum(log|diag(U)|)``.

    Parameters
    ----------
    M:
        Sparse symmetric positive-definite matrix in CSC format.

    Returns
    -------
    float  (natural log of the determinant)
    """
    lu = spla.splu(M)
    return float(np.sum(np.log(np.abs(lu.U.diagonal()))))


def _sparse_solve(M: sp.csc_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve M x = rhs where M is sparse SPD."""
    return np.asarray(spla.spsolve(M, rhs))


# ---------------------------------------------------------------------------
# Cross-product precomputation
# ---------------------------------------------------------------------------


def _precompute(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
) -> dict[str, np.ndarray | sp.csc_matrix | float]:
    """Precompute all cross-products that are constant across REML evaluations."""
    ZtZ: sp.csc_matrix = (Z.T @ Z).tocsc()
    ZtX: np.ndarray = (
        (Z.T @ X).toarray() if sp.issparse(Z.T @ X) else np.asarray(Z.T @ X)
    )
    Zty: np.ndarray = np.asarray(Z.T @ y).squeeze()
    XtX: np.ndarray = X.T @ X
    Xty: np.ndarray = X.T @ y
    yty: float = float(y @ y)
    return dict(ZtZ=ZtZ, ZtX=ZtX, Zty=Zty, XtX=XtX, Xty=Xty, yty=yty)


def _build_theta_bounds(
    specs: list[RandomEffectSpec],
) -> list[tuple[float | None, float | None]]:
    """Build L-BFGS-B bounds for the theta vector given a list of specs.

    Diagonal elements of L (positive definiteness) are bounded below at 1e-8.
    Off-diagonal elements (correlated only) are unconstrained.
    Independent-spec elements are all bounded below at 1e-8.
    """
    bounds: list[tuple[float | None, float | None]] = []
    for spec in specs:
        p_j = spec.n_terms
        if p_j == 1:
            bounds.append((1e-8, None))
        elif spec.correlated:
            for row in range(p_j):
                for col in range(row + 1):
                    bounds.append((1e-8, None) if row == col else (None, None))
        else:
            for _ in range(p_j):
                bounds.append((1e-8, None))
    return bounds


def _build_A11(
    ZtZ: sp.csc_matrix,
    lambda_diag_or_matrix: np.ndarray | sp.csc_matrix,
) -> sp.csc_matrix:
    """Build A11 = Lambda' Z'Z Lambda + I_q.

    Accepts either a 1-D diagonal vector (legacy path, fast element-wise
    scaling) or a full sparse Lambda matrix (generalised path using sparse
    matrix multiplication).
    """
    if sp.issparse(lambda_diag_or_matrix):
        Lambda = lambda_diag_or_matrix
        A = (Lambda.T @ ZtZ @ Lambda).tocsc()
        q = A.shape[0]
        return (A + sp.eye(q, format="csc")).tocsc()
    # Legacy diagonal path
    lambda_diag = lambda_diag_or_matrix
    coo = ZtZ.tocoo()
    scaled_data = coo.data * lambda_diag[coo.row] * lambda_diag[coo.col]
    q = ZtZ.shape[0]
    A11 = sp.csc_matrix((scaled_data, (coo.row, coo.col)), shape=(q, q)) + sp.eye(
        q, format="csc"
    )
    return A11


# ---------------------------------------------------------------------------
# Profiled REML objective
# ---------------------------------------------------------------------------


def reml_objective(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    _cache: dict[str, np.ndarray | sp.csc_matrix | float] | None = None,
    *,
    specs: list[RandomEffectSpec] | None = None,
    n_levels: list[int] | None = None,
) -> float:
    """Profiled REML deviance (to minimise over theta).

    Evaluates:

        obj(theta) = log|A11| + log|X'Omega^{-1}X| + (n-p) * log(y'Py)

    where constants (independent of theta) are dropped.

    Parameters
    ----------
    theta:
        Relative covariance parameters. When *specs* is ``None``: one scalar
        per grouping factor. When *specs* is provided: concatenated per-spec
        theta blocks as returned by :func:`make_lambda`.
    y, X, Z:
        Response, fixed-effects matrix, random-effects design matrix.
    q_sizes:
        Number of levels per grouping factor (legacy path; ignored when
        *specs* is provided).
    _cache:
        Optional dict of precomputed cross-products.
    specs:
        If provided, use :func:`make_lambda` to build a full sparse Lambda
        (supports random slopes). Otherwise fall back to the diagonal path.
    n_levels:
        Number of group levels per spec. Required when *specs* is not ``None``.

    Returns
    -------
    float  (profiled REML deviance; lower is better)
    """
    if _cache is None:
        _cache = _precompute(y, X, Z)

    ZtZ = sp.csc_matrix(_cache["ZtZ"])
    ZtX = np.asarray(_cache["ZtX"])
    Zty = np.asarray(_cache["Zty"])
    XtX = np.asarray(_cache["XtX"])
    Xty = np.asarray(_cache["Xty"])
    yty = float(_cache["yty"])  # noqa: PGH003

    n, p = X.shape

    # --- Build Lambda and A11 ---
    if specs is not None:
        Lambda = make_lambda(theta, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
    else:
        lambda_diag = make_lambda_diag(theta, q_sizes)
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty  # (q,)
        lZtX = lambda_diag[:, None] * ZtX  # (q, p)

    # --- Sparse Cholesky: prefer CHOLMOD (one numeric refactorisation) ---
    chol_factor = _cache.get("chol_factor") if _cache is not None else None
    if chol_factor is not None:
        chol_factor.cholesky(A11)  # type: ignore[union-attr]
        log_det_A11 = float(chol_factor.logdet())  # type: ignore[union-attr]
        c1 = np.asarray(chol_factor.solve_A(lZty)).squeeze()  # type: ignore[union-attr]
        C_X = np.asarray(chol_factor.solve_A(lZtX))  # type: ignore[union-attr]
    else:
        log_det_A11 = sparse_chol_logdet(A11)
        c1 = _sparse_solve(A11, lZty)  # (q,)
        C_X = _sparse_solve(A11, lZtX)  # (q, p)

    # --- X'Omega^{-1}X and X'Omega^{-1}y ---
    MX = XtX - lZtX.T @ C_X  # (p, p)
    rhs = Xty - lZtX.T @ c1  # (p,)

    # --- beta_hat and y'Py ---
    try:
        beta_hat = la.solve(MX, rhs, assume_a="pos")
    except la.LinAlgError:
        return np.inf

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    if yPy <= 0:
        return np.inf

    log_det_MX = float(np.linalg.slogdet(MX)[1])

    return float(log_det_A11 + log_det_MX + (n - p) * np.log(yPy))


# ---------------------------------------------------------------------------
# L-BFGS-B optimiser
# ---------------------------------------------------------------------------


def fit_reml(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    theta0: np.ndarray | None = None,
    *,
    specs: list[RandomEffectSpec] | None = None,
    n_levels: list[int] | None = None,
    optimizer: str = "lbfgsb",
) -> REMLResult:
    """Fit a linear mixed model by profiled REML.

    Parameters
    ----------
    y:        Response vector, shape (n,).
    X:        Fixed-effects design matrix, shape (n, p). Must include intercept.
    Z:        Joint random-effects design matrix, shape (n, q).
    q_sizes:  Number of levels for each grouping factor (legacy path; ignored
              when *specs* is provided).
    theta0:   Initial theta (defaults to ones of the appropriate length).
    specs:    Random effect specifications. When provided, uses
              :func:`make_lambda` to build a full block-diagonal Lambda
              (supports random slopes). ``n_levels`` must also be provided.
    n_levels: Number of group levels per spec.
    optimizer:
        ``"lbfgsb"`` (default) uses ``scipy.optimize.minimize`` with
        ``method="L-BFGS-B"``.  ``"bobyqa"`` uses ``pybobyqa`` (must be
        installed via the ``bobyqa`` optional extra), a gradient-free
        trust-region method that is more robust near variance-parameter
        boundaries and is the same algorithm used by lme4.

    Returns
    -------
    REMLResult
    """
    if optimizer not in ("lbfgsb", "bobyqa"):
        msg = f"optimizer must be 'lbfgsb' or 'bobyqa', got {optimizer!r}"
        raise ValueError(msg)

    n, p = X.shape

    if specs is not None:
        n_theta = sum(n_theta_for_spec(s.n_terms, s.correlated) for s in specs)
        bounds = _build_theta_bounds(specs)
    else:
        n_theta = len(q_sizes)
        bounds = [(1e-8, None)] * n_theta

    if theta0 is None:
        theta0 = np.ones(n_theta)

    cache = _precompute(y, X, Z)

    # Symbolic Cholesky analysis (once): sparsity pattern of A11 is fixed across
    # all theta evaluations, so only the numeric refactorisation is needed per call.
    cholmod = _try_cholmod()
    if cholmod is not None:
        if specs is not None:
            Lambda0 = make_lambda(theta0, specs, n_levels)  # type: ignore[arg-type]
            A11_0 = _build_A11(cache["ZtZ"], Lambda0)
        else:
            lambda_diag_0 = make_lambda_diag(theta0, q_sizes)
            A11_0 = _build_A11(cache["ZtZ"], lambda_diag_0)
        chol_factor = cholmod.analyze(A11_0)
        chol_factor.cholesky(A11_0)
        cache["chol_factor"] = chol_factor

    def obj(theta: np.ndarray) -> float:
        return reml_objective(
            theta, y, X, Z, q_sizes, _cache=cache, specs=specs, n_levels=n_levels
        )

    if optimizer == "bobyqa":
        import pybobyqa

        lower = np.array([lo if lo is not None else -np.inf for lo, _ in bounds])
        upper = np.array([hi if hi is not None else np.inf for _, hi in bounds])
        soln = pybobyqa.solve(obj, theta0, bounds=(lower, upper))
        theta_hat = soln.x
        converged = soln.msg == "Success: rho has reached rhoend"
    else:
        res = opt.minimize(obj, theta0, method="L-BFGS-B", bounds=bounds)
        theta_hat = res.x
        converged = bool(res.success)

    # --- Recover beta and sigma2 at optimum ---
    ZtZ = cache["ZtZ"]
    ZtX = cache["ZtX"]
    Zty = cache["Zty"]
    XtX = cache["XtX"]
    Xty = cache["Xty"]
    yty = cache["yty"]

    if specs is not None:
        Lambda = make_lambda(theta_hat, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
    else:
        lambda_diag = make_lambda_diag(theta_hat, q_sizes)
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty
        lZtX = lambda_diag[:, None] * ZtX

    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1
    beta_hat = la.solve(MX, rhs, assume_a="pos")

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    sigma2 = yPy / (n - p)

    # --- REML log-likelihood ---
    log_det_A11 = sparse_chol_logdet(A11)
    log_det_MX = float(np.linalg.slogdet(MX)[1])
    llf = -0.5 * (
        log_det_A11 + log_det_MX + (n - p) * (1.0 + np.log(2.0 * np.pi * sigma2))
    )

    # --- Information criteria ---
    nparams = p + n_theta + 1
    aic = -2.0 * llf + 2.0 * nparams
    bic = -2.0 * llf + np.log(n) * nparams

    return REMLResult(
        beta=beta_hat,
        theta=theta_hat,
        sigma2=sigma2,
        converged=converged,
        llf=float(llf),
        aic=float(aic),
        bic=float(bic),
        nobs=n,
        nparams=nparams,
        specs=specs,
        n_levels=n_levels,
    )
