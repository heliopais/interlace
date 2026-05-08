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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    from interlace.correlation import CorStruct
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
    fe_cov: np.ndarray | None = None  # sigma2 * (X'Ω⁻¹X)^{-1}
    _A11: Any = None  # A11 = I + W'W (q×q sparse) at optimum theta
    _W: Any = None  # W = Z @ Lambda (n×q sparse) at optimum theta
    correlation_params: dict[str, float] = field(default_factory=dict)


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


class LambdaBuilder:
    """Cached-pattern Lambda_theta builder.

    The non-zero pattern of Lambda_theta is determined by ``specs`` and
    ``n_levels`` alone — not by ``theta``.  Each non-zero entry of Lambda
    is exactly one component of theta (``make_lambda`` is linear in theta
    with coefficient 1), so we precompute (indices, indptr) once and a
    map from data-slot to theta-index, then write theta values into a
    fresh data array per ``update`` call.

    ~5x cheaper than calling ``make_lambda`` per outer optimiser step.
    Numerically equivalent to ``make_lambda(theta, specs, n_levels)``.
    """

    def __init__(
        self,
        specs: list[RandomEffectSpec],
        n_levels: list[int],
    ) -> None:
        self.specs = specs
        self.n_levels = list(n_levels)
        n_theta = sum(n_theta_for_spec(s.n_terms, s.correlated) for s in specs)
        if n_theta == 0:
            raise ValueError("LambdaBuilder requires at least one theta parameter")

        # Build a template Lambda where each theta component is a distinct
        # positive integer.  The resulting CSC's data array then encodes,
        # per slot, the (1-indexed) theta component it carries.
        theta_unit = np.arange(1, n_theta + 1, dtype=float)
        template = make_lambda(theta_unit, specs, n_levels)
        template.sort_indices()

        # Verify that every data slot is exactly one theta component
        # (no products, sums, or sign flips beyond what theta carries).
        data_int = np.rint(template.data).astype(np.intp)
        if (
            not np.allclose(template.data, data_int)
            or data_int.min() < 1
            or data_int.max() > n_theta
        ):
            msg = (
                "LambdaBuilder: make_lambda no longer linear-with-coeff-1 in "
                "theta; recipe needs updating"
            )
            raise RuntimeError(msg)

        self._indices = template.indices
        self._indptr = template.indptr
        self._shape = template.shape
        self._theta_map = data_int - 1  # 0-indexed
        self._n_theta = n_theta

        # Convenience: when every spec is scalar (p_j=1), Lambda is purely
        # diagonal with values theta_j repeated n_levels[j] times.  Inner-loop
        # callers can then bypass the sparse matmul Z @ Lambda entirely.
        self.is_diagonal = all(s.n_terms == 1 for s in specs)
        self.q: int = template.shape[0]

    def diag(self, theta: np.ndarray) -> np.ndarray:
        """Return Lambda's diagonal as a flat vector.  Only valid when
        ``is_diagonal`` is True."""
        if not self.is_diagonal:
            msg = "diag() only supported for scalar-only specs"
            raise ValueError(msg)
        return np.repeat(theta, self.n_levels)

    def update(self, theta: np.ndarray) -> sp.csc_matrix:
        """Return Lambda_theta as a CSC, sharing cached indices/indptr."""
        if theta.shape != (self._n_theta,):
            msg = f"theta shape {theta.shape} != ({self._n_theta},)"
            raise ValueError(msg)
        data = theta[self._theta_map]
        m = sp.csc_matrix(
            (data, self._indices, self._indptr),
            shape=self._shape,
            copy=False,
        )
        # scipy's constructor may copy indices for dtype safety; restore
        # the cached buffers so successive callers see the same arrays.
        m.indices = self._indices
        m.indptr = self._indptr
        return m


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
        from sksparse import cholmod  # type: ignore[import-untyped]

        return cholmod
    except ImportError:
        return None


def _init_chol_factor(cholmod_mod: Any, A11: sp.csc_matrix) -> tuple[Any, str | None]:
    """Initialise a CHOLMOD factor, handling both sksparse API versions.

    sksparse >= 0.5.0 changed the public API:
      - New: ``cho_factor(A)`` → ``CholeskyFactor`` with ``.factorize()``,
        ``.logdet()``, ``.solve(b, 'A')`` methods.
      - Old: ``cholesky(A)`` → ``Factor`` with ``.cholesky()``,
        ``.logdet()``, ``.solve_A(b)`` methods.

    Returns ``(factor, api)`` where *api* is ``"new"``, ``"old"``, or
    ``None`` on failure.
    """
    # Try new API first (sksparse >= 0.5.0)
    if hasattr(cholmod_mod, "cho_factor"):
        try:
            factor = cholmod_mod.cho_factor(A11)
            if (
                hasattr(factor, "factorize")
                and hasattr(factor, "logdet")
                and hasattr(factor, "solve")
            ):
                return factor, "new"
        except Exception:  # noqa: BLE001
            pass
    # Fall back to old API (sksparse < 0.5.0)
    try:
        factor = cholmod_mod.cholesky(A11)
        if (
            hasattr(factor, "cholesky")
            and hasattr(factor, "logdet")
            and hasattr(factor, "solve_A")
        ):
            return factor, "old"
    except Exception:  # noqa: BLE001
        pass
    return None, None


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
    """Solve M x = rhs where M is sparse SPD.

    Preserves the 2D shape of *rhs*: ``spla.spsolve`` silently squeezes a
    ``(q, 1)`` rhs to ``(q,)``, which breaks downstream matmul shape contracts
    (e.g. ``C_X @ beta_hat`` in :func:`reml_gradient` for intercept-only X).
    """
    out = np.asarray(spla.spsolve(M, rhs))
    if rhs.ndim == 2 and out.ndim == 1:
        out = out.reshape(rhs.shape[0], -1)
    return out


# ---------------------------------------------------------------------------
# Cross-product precomputation
# ---------------------------------------------------------------------------


def _precompute(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray | sp.csc_matrix | float]:
    """Precompute all cross-products that are constant across REML evaluations.

    When *weights* is provided, all cross-products are weighted:
    ``X'WX``, ``X'Wy``, ``Z'WZ``, ``Z'WX``, ``Z'Wy``, ``y'Wy``
    where ``W = diag(weights)``.  This is equivalent to pre-multiplying all
    data vectors/matrices by ``sqrt(W)``.
    """
    if weights is not None:
        sqW = np.sqrt(weights)
        Xw = sqW[:, None] * X
        yw = sqW * y
        Zw = sp.diags(sqW, format="csc") @ Z
    else:
        Xw = X
        yw = y
        Zw = Z

    ZtZ: sp.csc_matrix = (Zw.T @ Zw).tocsc()
    ZtX: np.ndarray = (
        (Zw.T @ Xw).toarray() if sp.issparse(Zw.T @ Xw) else np.asarray(Zw.T @ Xw)
    )
    Zty: np.ndarray = np.asarray(Zw.T @ yw).squeeze()
    XtX: np.ndarray = Xw.T @ Xw
    Xty: np.ndarray = Xw.T @ yw
    yty: float = float(yw @ yw)
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

# Below this q the dense kernel beats the sparse one (interlace-72oc): the
# constant overhead of scipy.sparse construction dwarfs the actual
# factorisation work at small q.
_DENSE_Q_THRESHOLD = 200


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

    Dispatches to a dense kernel for small q (interlace-72oc); otherwise
    falls through to the sparse path.  The sparse and dense kernels are
    numerically equivalent.

    See :func:`reml_objective_sparse` / :func:`reml_objective_dense` for the
    underlying implementations and the original docstring text.
    """
    q = Z.shape[1]
    if q <= _DENSE_Q_THRESHOLD:
        return reml_objective_dense(
            theta,
            y,
            X,
            Z,
            q_sizes,
            _cache=_cache,
            specs=specs,
            n_levels=n_levels,
        )
    return reml_objective_sparse(
        theta,
        y,
        X,
        Z,
        q_sizes,
        _cache=_cache,
        specs=specs,
        n_levels=n_levels,
    )


def reml_objective_sparse(
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
    """Sparse profiled REML deviance.  The original implementation; preserved
    for the q > _DENSE_Q_THRESHOLD path."""
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
    if specs is not None and not all(s.n_terms == 1 for s in specs):
        # General path: random slopes require full sparse Lambda
        builder = _cache.get("lambda_builder") if _cache is not None else None
        if builder is not None:
            Lambda = builder.update(theta)  # type: ignore[union-attr]
        else:
            Lambda = make_lambda(theta, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
    else:
        # Fast diagonal path: intercept-only specs (p_j=1 for all j)
        _q_sizes = n_levels if specs is not None else q_sizes
        lambda_diag = make_lambda_diag(theta, _q_sizes)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty  # (q,)
        lZtX = lambda_diag[:, None] * ZtX  # (q, p)

    # --- Sparse Cholesky: prefer CHOLMOD (one numeric refactorisation) ---
    chol_factor = _cache.get("chol_factor") if _cache is not None else None
    chol_api = _cache.get("chol_api", "old") if _cache is not None else "old"
    if chol_factor is not None:
        if chol_api == "new":
            chol_factor.factorize(A11)  # type: ignore[union-attr]
            log_det_A11 = float(chol_factor.logdet())  # type: ignore[union-attr]
            c1 = np.asarray(chol_factor.solve(lZty, "A")).squeeze()  # type: ignore[union-attr]
            C_X = np.asarray(chol_factor.solve(lZtX, "A"))  # type: ignore[union-attr]
        else:
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


def reml_objective_dense(
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
    """Dense profiled REML deviance for small q (interlace-72oc).

    Mirrors :func:`reml_objective_sparse` line-for-line but keeps ZtZ, the
    Lambda factor, and A11 as plain numpy arrays.  ``ZtZ_dense`` is cached
    in the ``_cache`` dict so it materialises once per fit.
    """
    if _cache is None:
        _cache = _precompute(y, X, Z)

    ZtX = np.asarray(_cache["ZtX"])
    Zty = np.asarray(_cache["Zty"])
    XtX = np.asarray(_cache["XtX"])
    Xty = np.asarray(_cache["Xty"])
    yty = float(_cache["yty"])  # noqa: PGH003

    ZtZ_dense_cached = _cache.get("ZtZ_dense") if _cache is not None else None
    if ZtZ_dense_cached is None:
        ZtZ_sp = _cache["ZtZ"]
        ZtZ_dense = (
            ZtZ_sp.toarray()  # type: ignore[union-attr]
            if sp.issparse(ZtZ_sp)
            else np.asarray(ZtZ_sp)
        )
        _cache["ZtZ_dense"] = ZtZ_dense
    else:
        ZtZ_dense = np.asarray(ZtZ_dense_cached)

    n, p = X.shape
    q = ZtZ_dense.shape[0]
    eye_q = np.eye(q)

    # --- Build Lambda and A11_dense ---
    if specs is not None and not all(s.n_terms == 1 for s in specs):
        builder = _cache.get("lambda_builder") if _cache is not None else None
        if builder is not None:
            Lambda_dense = np.asarray(builder.update(theta).toarray())  # type: ignore[union-attr]
        else:
            Lambda_dense = np.asarray(
                make_lambda(theta, specs, n_levels).toarray()  # type: ignore[arg-type]
            )
        A11 = Lambda_dense.T @ ZtZ_dense @ Lambda_dense + eye_q
        lZty = Lambda_dense.T @ Zty
        lZtX = Lambda_dense.T @ ZtX
    else:
        _q_sizes = n_levels if specs is not None else q_sizes
        lambda_diag = make_lambda_diag(theta, _q_sizes)  # type: ignore[arg-type]
        A11 = lambda_diag[:, None] * ZtZ_dense * lambda_diag[None, :] + eye_q
        lZty = lambda_diag * Zty
        lZtX = lambda_diag[:, None] * ZtX

    # --- Dense Cholesky ---
    try:
        L = np.linalg.cholesky(A11)
    except np.linalg.LinAlgError:
        return np.inf
    log_det_A11 = 2.0 * float(np.sum(np.log(np.diag(L))))

    c1 = la.cho_solve((L, True), lZty)
    C_X = la.cho_solve((L, True), lZtX)

    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1
    try:
        beta_hat = la.solve(MX, rhs, assume_a="pos")
    except la.LinAlgError:
        return np.inf

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    if yPy <= 0:
        return np.inf

    log_det_MX = float(np.linalg.slogdet(MX)[1])
    return float(log_det_A11 + log_det_MX + (n - p) * np.log(yPy))


def reml_gradient(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    _cache: dict[str, np.ndarray | sp.csc_matrix | float] | None = None,
    *,
    specs: list[RandomEffectSpec] | None = None,
    n_levels: list[int] | None = None,
) -> np.ndarray:
    """Analytical gradient of the profiled REML deviance w.r.t. theta.

    Dispatches to :func:`reml_gradient_dense` for small q (interlace-72oc),
    otherwise to :func:`reml_gradient_sparse`.  Only supported for the
    diagonal (intercept-only) path -- raises ``NotImplementedError`` for
    random-slope specs.
    """
    if specs is not None and not all(s.n_terms == 1 for s in specs):
        msg = "reml_gradient is only implemented for the diagonal (intercept-only) path"
        raise NotImplementedError(msg)
    q = Z.shape[1]
    if q <= _DENSE_Q_THRESHOLD:
        return reml_gradient_dense(
            theta,
            y,
            X,
            Z,
            q_sizes,
            _cache=_cache,
            specs=specs,
            n_levels=n_levels,
        )
    return reml_gradient_sparse(
        theta,
        y,
        X,
        Z,
        q_sizes,
        _cache=_cache,
        specs=specs,
        n_levels=n_levels,
    )


def reml_gradient_sparse(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    _cache: dict[str, np.ndarray | sp.csc_matrix | float] | None = None,
    *,
    specs: list[RandomEffectSpec] | None = None,
    n_levels: list[int] | None = None,
) -> np.ndarray:
    """Sparse analytical gradient.  Original implementation; preserved for
    the q > _DENSE_Q_THRESHOLD path.

    The gradient decomposes into three terms per theta_k:

        d obj / d theta_k
            = tr(A11^{-1} dA11_k)               [log|A11| term]
            + tr(MX^{-1} dMX_k)                 [log|MX|  term]
            + (n-p)/yPy * d(yPy)/d theta_k      [log(yPy) term]
    """
    if specs is not None and not all(s.n_terms == 1 for s in specs):
        msg = "reml_gradient is only implemented for the diagonal (intercept-only) path"
        raise NotImplementedError(msg)

    if _cache is None:
        _cache = _precompute(y, X, Z)

    ZtZ = sp.csc_matrix(_cache["ZtZ"])
    ZtX = np.asarray(_cache["ZtX"])
    Zty = np.asarray(_cache["Zty"])
    XtX = np.asarray(_cache["XtX"])
    Xty = np.asarray(_cache["Xty"])
    yty = float(_cache["yty"])

    _q_sizes: list[int] = n_levels if specs is not None else q_sizes  # type: ignore[assignment]
    lambda_diag = make_lambda_diag(theta, _q_sizes)
    A11 = _build_A11(ZtZ, lambda_diag)
    lZty = lambda_diag * Zty
    lZtX = lambda_diag[:, None] * ZtX

    n, p = X.shape
    q = A11.shape[0]

    # Single factorisation reused for c1, C_X, and the dense inverse.  When
    # the caller passed a primed cholmod factor in _cache (mirrors the path
    # taken by reml_objective), reuse it; otherwise fall back to a single
    # SuperLU factorisation.
    chol_factor = _cache.get("chol_factor") if _cache is not None else None
    chol_api = _cache.get("chol_api", "old") if _cache is not None else "old"
    eye_q = np.eye(q)
    if chol_factor is not None:
        if chol_api == "new":
            chol_factor.factorize(A11)  # type: ignore[union-attr]
            c1 = np.asarray(chol_factor.solve(lZty, "A")).squeeze()  # type: ignore[union-attr]
            C_X = np.asarray(chol_factor.solve(lZtX, "A"))  # type: ignore[union-attr]
            A11_inv = np.asarray(chol_factor.solve(eye_q, "A"))  # type: ignore[union-attr]
        else:
            chol_factor.cholesky(A11)  # type: ignore[union-attr]
            c1 = np.asarray(chol_factor.solve_A(lZty)).squeeze()  # type: ignore[union-attr]
            C_X = np.asarray(chol_factor.solve_A(lZtX))  # type: ignore[union-attr]
            A11_inv = np.asarray(chol_factor.solve_A(eye_q))  # type: ignore[union-attr]
    else:
        # SuperLU fallback: factorise A11 once, reuse the LU factor for all
        # three solves (c1, C_X, dense inverse).  Saves two redundant
        # factorisations vs the previous code path.
        lu = spla.splu(A11)
        c1 = np.asarray(lu.solve(lZty))
        C_X = np.asarray(lu.solve(lZtX))
        A11_inv = lu.solve(eye_q)
    if c1.ndim == 2 and c1.shape[1] == 1:
        c1 = c1.ravel()

    MX = XtX - lZtX.T @ C_X  # (p, p)
    rhs = Xty - lZtX.T @ c1  # (p,)
    beta_hat = la.solve(MX, rhs, assume_a="pos")
    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    MX_inv = np.linalg.inv(MX)  # (p, p) small dense inverse

    # Shared quantities across factors
    coo = ZtZ.tocoo()
    ZtZ_csr = ZtZ.tocsr()
    f = c1 - C_X @ beta_hat  # "BLUP residual" in RE space (q,)
    lf = lambda_diag * f  # element-wise (q,)
    Zt_resid = Zty - ZtX @ beta_hat  # Z'(y - X beta_hat) (q,)

    grad = np.zeros(len(theta))
    q_start = 0
    for k, q_k in enumerate(_q_sizes):
        q_end = q_start + q_k

        # dA11/dtheta_k data in the COO sparsity pattern of ZtZ:
        # dA11[i,j] = ZtZ[i,j] * (e_k[i]*lambda[j] + lambda[i]*e_k[j])
        ek_row = (coo.row >= q_start) & (coo.row < q_end)
        ek_col = (coo.col >= q_start) & (coo.col < q_end)
        dA11_data = coo.data * (
            ek_row.astype(float) * lambda_diag[coo.col]
            + lambda_diag[coo.row] * ek_col.astype(float)
        )

        # Term 1: tr(A11^{-1} dA11_k)
        # Both matrices symmetric → tr(A·B) = frobenius(A, B) = sum A[i,j]*B[i,j]
        term1 = float(np.sum(A11_inv[coo.row, coo.col] * dA11_data))

        # Term 2: tr(MX^{-1} dMX_k)
        # dMX_k = -dB_k' C_X - C_X' dB_k + C_X' dA11_k C_X
        # tr(MX^{-1} dMX_k) = -2 tr(MX^{-1} C_X' dB_k) + tr(MX^{-1} C_X' dA11_k C_X)
        # where dB_k = e_k[:,None]*ZtX  →  C_X'dB_k = C_X[idx_k,:]' ZtX[idx_k,:]
        C_XT_dBk = C_X[q_start:q_end, :].T @ ZtX[q_start:q_end, :]  # (p, p)
        dA11_sp = sp.csc_matrix((dA11_data, (coo.row, coo.col)), shape=(q, q))
        dA_CX = dA11_sp @ C_X  # (q, p)
        C_XT_dA_CX = C_X.T @ dA_CX  # (p, p)
        term2 = float(
            -2.0 * np.trace(MX_inv @ C_XT_dBk) + np.trace(MX_inv @ C_XT_dA_CX)
        )

        # Term 3: (n-p)/yPy · d(yPy)/dtheta_k
        # d(yPy)/dtheta_k = 2 f[idx_k] · (ZtZ[idx_k,:] lf  −  Zt_resid[idx_k])
        ZtZ_k_lf = np.asarray(ZtZ_csr[q_start:q_end, :] @ lf).ravel()
        d_yPy = float(2.0 * f[q_start:q_end] @ (ZtZ_k_lf - Zt_resid[q_start:q_end]))
        term3 = float((n - p) / yPy * d_yPy)

        grad[k] = term1 + term2 + term3
        q_start = q_end

    return grad


def reml_gradient_dense(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    _cache: dict[str, np.ndarray | sp.csc_matrix | float] | None = None,
    *,
    specs: list[RandomEffectSpec] | None = None,
    n_levels: list[int] | None = None,
) -> np.ndarray:
    """Dense analytical gradient for small q (interlace-72oc).

    Mirrors :func:`reml_gradient_sparse` but with all matrices kept dense.
    Diagonal (intercept-only) only.
    """
    if specs is not None and not all(s.n_terms == 1 for s in specs):
        msg = "reml_gradient is only implemented for the diagonal (intercept-only) path"
        raise NotImplementedError(msg)

    if _cache is None:
        _cache = _precompute(y, X, Z)

    ZtX = np.asarray(_cache["ZtX"])
    Zty = np.asarray(_cache["Zty"])
    XtX = np.asarray(_cache["XtX"])
    Xty = np.asarray(_cache["Xty"])
    yty = float(_cache["yty"])

    ZtZ_dense_cached = _cache.get("ZtZ_dense") if _cache is not None else None
    if ZtZ_dense_cached is None:
        ZtZ_sp = _cache["ZtZ"]
        ZtZ_dense = (
            ZtZ_sp.toarray()  # type: ignore[union-attr]
            if sp.issparse(ZtZ_sp)
            else np.asarray(ZtZ_sp)
        )
        _cache["ZtZ_dense"] = ZtZ_dense
    else:
        ZtZ_dense = np.asarray(ZtZ_dense_cached)

    n, p = X.shape
    q = ZtZ_dense.shape[0]
    eye_q = np.eye(q)

    _q_sizes: list[int] = n_levels if specs is not None else q_sizes  # type: ignore[assignment]
    lambda_diag = make_lambda_diag(theta, _q_sizes)
    A11 = lambda_diag[:, None] * ZtZ_dense * lambda_diag[None, :] + eye_q
    lZty = lambda_diag * Zty
    lZtX = lambda_diag[:, None] * ZtX

    L = np.linalg.cholesky(A11)
    c1 = la.cho_solve((L, True), lZty)
    C_X = la.cho_solve((L, True), lZtX)
    A11_inv = la.cho_solve((L, True), eye_q)

    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1
    beta_hat = la.solve(MX, rhs, assume_a="pos")
    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    MX_inv = np.linalg.inv(MX)

    f = c1 - C_X @ beta_hat
    lf = lambda_diag * f
    Zt_resid = Zty - ZtX @ beta_hat

    grad = np.zeros(len(theta))
    q_start = 0
    for k, q_k in enumerate(_q_sizes):
        q_end = q_start + q_k
        # dA11_k[i,j] = ZtZ[i,j] * (e_k[i]*lambda[j] + lambda[i]*e_k[j])
        ek = np.zeros(q)
        ek[q_start:q_end] = 1.0
        scale = np.outer(ek, lambda_diag) + np.outer(lambda_diag, ek)
        dA11 = ZtZ_dense * scale  # (q, q)

        term1 = float(np.sum(A11_inv * dA11))

        # dMX_k = -dB_k' C_X - C_X' dB_k + C_X' dA11_k C_X
        C_XT_dBk = C_X[q_start:q_end, :].T @ ZtX[q_start:q_end, :]
        dA_CX = dA11 @ C_X
        C_XT_dA_CX = C_X.T @ dA_CX
        term2 = float(
            -2.0 * np.trace(MX_inv @ C_XT_dBk) + np.trace(MX_inv @ C_XT_dA_CX)
        )

        ZtZ_k_lf = ZtZ_dense[q_start:q_end, :] @ lf
        d_yPy = float(2.0 * f[q_start:q_end] @ (ZtZ_k_lf - Zt_resid[q_start:q_end]))
        term3 = float((n - p) / yPy * d_yPy)

        grad[k] = term1 + term2 + term3
        q_start = q_end

    return grad


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
    tight: bool = True,
    use_gradient: bool | None = None,
    weights: np.ndarray | None = None,
    correlation: CorStruct | None = None,
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
    weights:
        Observation-level prior weights, shape (n,). Defaults to ones.
    correlation:
        Residual correlation structure (e.g. ``AR1("time")``).  Must have
        ``setup()`` already called.  When provided, the correlation parameter(s)
        are jointly estimated with theta.

    Returns
    -------
    REMLResult
    """
    if optimizer not in ("lbfgsb", "bobyqa", "nelder-mead"):
        msg = (
            f"optimizer must be 'lbfgsb', 'bobyqa', or 'nelder-mead', got {optimizer!r}"
        )
        raise ValueError(msg)

    n, p = X.shape

    if specs is not None:
        n_theta = sum(n_theta_for_spec(s.n_terms, s.correlated) for s in specs)
        bounds = _build_theta_bounds(specs)
    else:
        n_theta = len(q_sizes)
        bounds = [(1e-8, None)] * n_theta

    # Auto-resolve use_gradient.  Analytic gradient is only implemented for
    # the diagonal (intercept-only) path; with random slopes it always falls
    # back to forward-difference.  Explicit ``True`` with slopes raises early.
    #
    # Default (use_gradient=None) is currently conservative (False): Phase A
    # of interlace-mxzk made the gradient call ~2x faster, but L-BFGS-B's FD
    # path remains competitive at q >= ~35 because the objective is cheap.
    # The default flip waits on a faster selected-inverse (Phase B / numba).
    is_diagonal = specs is None or all(s.n_terms == 1 for s in specs)
    if use_gradient is None:
        use_gradient = False
    elif use_gradient and not is_diagonal:
        msg = (
            "reml_gradient is only implemented for the diagonal "
            "(intercept-only) path; pass use_gradient=False or None for "
            "specs with random slopes."
        )
        raise NotImplementedError(msg)

    if theta0 is None:
        theta0 = np.ones(n_theta)

    # --- Correlation structure: joint optimisation over (theta, rho_raw) ---
    if correlation is not None:
        return _fit_reml_with_correlation(
            y,
            X,
            Z,
            q_sizes,
            theta0,
            specs=specs,
            n_levels=n_levels,
            optimizer=optimizer,
            tight=tight,
            weights=weights,
            correlation=correlation,
            n_theta=n_theta,
            bounds=bounds,
        )

    cache = _precompute(y, X, Z, weights=weights)

    # Cache structural sparse pattern of Lambda_theta when random slopes
    # are involved.  Only .data changes across theta evaluations.
    if specs is not None and not all(s.n_terms == 1 for s in specs):
        cache["lambda_builder"] = LambdaBuilder(specs, n_levels)  # type: ignore[arg-type]

    # Cholesky factorisation (once for sparsity analysis + initial numeric factor):
    # sparsity pattern of A11 is fixed across all theta evaluations, so only the
    # numeric refactorisation is needed per call (factor.cholesky reuses the pattern).
    cholmod = _try_cholmod()
    if cholmod is not None:
        if specs is not None and not all(s.n_terms == 1 for s in specs):
            Lambda0 = cache["lambda_builder"].update(theta0)  # type: ignore[union-attr]
            A11_0 = _build_A11(cache["ZtZ"], Lambda0)
        else:
            _q_init = n_levels if specs is not None else q_sizes
            lambda_diag_0 = make_lambda_diag(theta0, _q_init)  # type: ignore[arg-type]
            A11_0 = _build_A11(cache["ZtZ"], lambda_diag_0)
        factor, api = _init_chol_factor(cholmod, A11_0)
        if factor is not None:
            cache["chol_factor"] = factor
            cache["chol_api"] = api

    lower_bounds = np.array([lo if lo is not None else -np.inf for lo, _ in bounds])

    def obj(theta: np.ndarray) -> float:
        return reml_objective(
            theta, y, X, Z, q_sizes, _cache=cache, specs=specs, n_levels=n_levels
        )

    def grad(theta: np.ndarray) -> np.ndarray:
        return reml_gradient(
            theta, y, X, Z, q_sizes, _cache=cache, specs=specs, n_levels=n_levels
        )

    if optimizer == "bobyqa":
        import pybobyqa

        upper = np.array([hi if hi is not None else np.inf for _, hi in bounds])
        soln = pybobyqa.solve(obj, theta0, bounds=(lower_bounds, upper))
        theta_hat = soln.x
        converged = soln.msg == "Success: rho has reached rhoend"
    elif optimizer == "nelder-mead":
        # Nelder-Mead has no native bound support; enforce lower bounds by
        # projecting theta onto the feasible region inside the objective.
        def obj_bounded(theta: np.ndarray) -> float:
            return obj(np.maximum(theta, lower_bounds))

        res = opt.minimize(obj_bounded, theta0, method="Nelder-Mead")
        theta_hat = np.maximum(res.x, lower_bounds)
        converged = bool(res.success)
    elif not tight and n_theta == 1:
        # Fast path for warm-started 1-D case-deletion refits: Brent's method
        # uses ~14 evals vs ~28 for L-BFGS-B, roughly 2× faster per refit.
        lo = lower_bounds[0]
        hi = bounds[0][1] if bounds[0][1] is not None else np.inf
        # Upper bound: search up to max(100 * theta0, 100) to avoid missing optimum
        hi_search = min(hi, max(100.0 * float(theta0[0]), 100.0))
        res_1d = opt.minimize_scalar(
            lambda t: obj(np.array([t])),  # noqa: E731
            bounds=(lo, hi_search),
            method="bounded",
        )
        theta_hat = np.array([res_1d.x])
        converged = bool(res_1d.success)
    else:
        # tight=False: limit iterations for warm-started case-deletion refits.
        # maxiter=10 caps expensive outlier refits; maxls=5 limits line-search evals.
        # Together they give ~2× speedup vs default convergence on large n.
        lbfgsb_opts = None if tight else {"maxiter": 10, "maxls": 5}
        jac = grad if use_gradient else None
        res = opt.minimize(
            obj, theta0, method="L-BFGS-B", bounds=bounds, jac=jac, options=lbfgsb_opts
        )
        theta_hat = res.x
        converged = bool(res.success)

    # --- Recover beta and sigma2 at optimum ---
    ZtZ = cache["ZtZ"]
    ZtX = cache["ZtX"]
    Zty = cache["Zty"]
    XtX = cache["XtX"]
    Xty = cache["Xty"]
    yty = cache["yty"]

    if specs is not None and not all(s.n_terms == 1 for s in specs):
        Lambda = make_lambda(theta_hat, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
        W_final: sp.csc_matrix = (Z @ Lambda).tocsc()
    else:
        _q_hat = n_levels if specs is not None else q_sizes
        lambda_diag = make_lambda_diag(theta_hat, _q_hat)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty
        lZtX = lambda_diag[:, None] * ZtX
        W_final = (Z @ sp.diags(lambda_diag, format="csc")).tocsc()

    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1
    beta_hat = la.solve(MX, rhs, assume_a="pos")

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    sigma2 = yPy / (n - p)
    fe_cov = sigma2 * np.linalg.inv(MX)

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
        fe_cov=fe_cov,
        _A11=A11,
        _W=W_final,
    )


# ---------------------------------------------------------------------------
# REML with residual correlation structure
# ---------------------------------------------------------------------------


def _fit_reml_with_correlation(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    theta0: np.ndarray,
    *,
    specs: list[RandomEffectSpec] | None = None,
    n_levels: list[int] | None = None,
    optimizer: str = "lbfgsb",
    tight: bool = True,
    weights: np.ndarray | None = None,
    correlation: CorStruct,
    n_theta: int,
    bounds: list[tuple[float | None, float | None]],
) -> REMLResult:
    """Fit REML with a residual correlation structure.

    Joint optimisation over (theta_RE, rho_raw) where rho_raw is the
    unconstrained parameterisation of the correlation parameter(s).
    At each evaluation, data is whitened by R^{-1/2}(rho) and the standard
    profiled REML is computed on the whitened data, plus the log|R| correction.
    """
    from interlace.correlation import rho_from_unconstrained, unconstrained_from_rho

    n, p = X.shape
    n_corr = correlation.n_corr_params

    # Initial rho_raw (unconstrained space); start near 0.3
    rho_raw0 = np.array([unconstrained_from_rho(0.3)] * n_corr)

    # Apply weights upfront (whitening commutes with weight scaling)
    if weights is not None:
        sqW = np.sqrt(weights)
        y_w0 = sqW * y
        X_w0 = sqW[:, None] * X
        Z_w0 = sp.diags(sqW, format="csc") @ Z
    else:
        y_w0 = y
        X_w0 = X
        Z_w0 = Z

    def obj_joint(params: np.ndarray) -> float:
        theta_re = params[:n_theta]
        rho_raw = params[n_theta:]
        rho = np.array([rho_from_unconstrained(float(r)) for r in rho_raw])

        # Whiten data
        y_w, X_w, Z_w = correlation.whiten_data(y_w0, X_w0, Z_w0, rho)

        # Compute standard REML objective on whitened data (no caching —
        # cross-products change with rho)
        cache_w = _precompute(y_w, X_w, Z_w)
        val = reml_objective(
            theta_re,
            y_w,
            X_w,
            Z_w,
            q_sizes,
            _cache=cache_w,
            specs=specs,
            n_levels=n_levels,
        )
        if not np.isfinite(val):
            return np.inf

        # Add log|R(rho)| correction
        log_det_r = correlation.log_det_R(rho)
        return float(val + log_det_r)

    # Joint initial params and bounds
    params0 = np.concatenate([theta0, rho_raw0])
    # RE theta bounds + correlation-specific bounds on rho_raw
    corr_bounds = correlation.unconstrained_bounds()
    bounds_joint = list(bounds) + corr_bounds

    lower_bounds_joint = np.array(
        [lo if lo is not None else -np.inf for lo, _ in bounds_joint]
    )

    if optimizer == "bobyqa":
        import pybobyqa

        upper = np.array([hi if hi is not None else np.inf for _, hi in bounds_joint])
        soln = pybobyqa.solve(obj_joint, params0, bounds=(lower_bounds_joint, upper))
        params_hat = soln.x
        converged = soln.msg == "Success: rho has reached rhoend"
    elif optimizer == "nelder-mead":

        def obj_bounded(params: np.ndarray) -> float:
            return obj_joint(np.maximum(params, lower_bounds_joint))

        res = opt.minimize(obj_bounded, params0, method="Nelder-Mead")
        params_hat = np.maximum(res.x, lower_bounds_joint)
        converged = bool(res.success)
    else:
        lbfgsb_opts = None if tight else {"maxiter": 10, "maxls": 5}
        res = opt.minimize(
            obj_joint,
            params0,
            method="L-BFGS-B",
            bounds=bounds_joint,
            options=lbfgsb_opts,
        )
        params_hat = res.x
        converged = bool(res.success)

    # --- Extract optimum theta_RE and rho ---
    theta_hat = params_hat[:n_theta]
    rho_raw_hat = params_hat[n_theta:]
    rho_hat = np.array([rho_from_unconstrained(float(r)) for r in rho_raw_hat])

    # Build correlation_params dict
    corr_params: dict[str, float] = {"rho": float(rho_hat[0])}

    # --- Recover beta and sigma2 at optimum on whitened data ---
    y_w, X_w, Z_w = correlation.whiten_data(y_w0, X_w0, Z_w0, rho_hat)
    cache_w = _precompute(y_w, X_w, Z_w)

    ZtZ = sp.csc_matrix(cache_w["ZtZ"])
    ZtX = np.asarray(cache_w["ZtX"])
    Zty = np.asarray(cache_w["Zty"])
    XtX = np.asarray(cache_w["XtX"])
    Xty = np.asarray(cache_w["Xty"])
    yty = float(cache_w["yty"])

    if specs is not None and not all(s.n_terms == 1 for s in specs):
        Lambda = make_lambda(theta_hat, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
        W_final: sp.csc_matrix = (Z_w @ Lambda).tocsc()
    else:
        _q_hat = n_levels if specs is not None else q_sizes
        lambda_diag = make_lambda_diag(theta_hat, _q_hat)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty
        lZtX = lambda_diag[:, None] * ZtX
        W_final = (Z_w @ sp.diags(lambda_diag, format="csc")).tocsc()

    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1
    beta_hat = la.solve(MX, rhs, assume_a="pos")

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    sigma2 = yPy / (n - p)
    fe_cov = sigma2 * np.linalg.inv(MX)

    # --- REML log-likelihood (includes log|R| correction) ---
    log_det_A11 = sparse_chol_logdet(A11)
    log_det_MX = float(np.linalg.slogdet(MX)[1])
    log_det_r = correlation.log_det_R(rho_hat)
    llf = -0.5 * (
        log_det_A11
        + log_det_r
        + log_det_MX
        + (n - p) * (1.0 + np.log(2.0 * np.pi * sigma2))
    )

    # nparams includes correlation parameter(s)
    nparams = p + n_theta + n_corr + 1
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
        fe_cov=fe_cov,
        _A11=A11,
        _W=W_final,
        correlation_params=corr_params,
    )


# ---------------------------------------------------------------------------
# Profiled ML objective
# ---------------------------------------------------------------------------


def profile_loglik(
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
    """Profile ML log-likelihood L(theta) at the profiled beta and sigma^2.

    For a given theta, beta and sigma^2 are set to their MLEs:

        beta_hat(theta) = (X' Omega^{-1} X)^{-1} X' Omega^{-1} y
        sigma2_hat(theta) = y' P(theta) y / n

    and the resulting log-likelihood is returned:

        L(theta) = -n/2 * (1 + log(2*pi*sigma2_hat)) - 1/2 * log|A11(theta)|

    Equivalently:

        L(theta) = -n/2 * (1 + log(2*pi/n)) - ml_objective(theta) / 2

    This is the foundation for profile confidence intervals: the CI endpoints
    are the theta values where 2*(L(theta_hat) - L(theta)) = chi2(level, df=1).

    Parameters
    ----------
    theta, y, X, Z, q_sizes, _cache, specs, n_levels:
        Same as :func:`ml_objective`.

    Returns
    -------
    float
        Profile log-likelihood at *theta*.  Returns ``-inf`` when the
        objective is non-finite (e.g. invalid theta).
    """
    n = len(y)
    deviance = ml_objective(
        theta, y, X, Z, q_sizes, _cache, specs=specs, n_levels=n_levels
    )
    if not np.isfinite(deviance):
        return -np.inf
    constant = -n / 2.0 * (1.0 + np.log(2.0 * np.pi / n))
    return float(constant - deviance / 2.0)


def _sigma2_at_theta(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    _cache: dict[str, Any] | None = None,
    *,
    specs: Any | None = None,
    n_levels: list[int] | None = None,
) -> float:
    """Return the ML-profiled residual variance sigma² at the given theta.

    Used by profile_confint to transform CI endpoints from theta scale to
    the natural SD scale: sd_b = theta_diag * sqrt(sigma2_at_theta(theta_boundary)).
    sigma² = y'P(theta)y / n, where beta is profiled out at each theta.
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

    if specs is not None and not all(s.n_terms == 1 for s in specs):
        builder = _cache.get("lambda_builder") if _cache is not None else None
        if builder is not None:
            Lambda = builder.update(theta)
        else:
            Lambda = make_lambda(theta, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
    else:
        _q_sizes: list[int] = n_levels if specs is not None else []  # type: ignore[assignment]
        lambda_diag = make_lambda_diag(theta, _q_sizes)
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty
        lZtX = lambda_diag[:, None] * ZtX

    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1

    try:
        beta_hat = la.solve(MX, rhs, assume_a="pos")
    except la.LinAlgError:
        return float("inf")

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    return float(max(yPy, 0.0) / int(n))


def ml_objective(
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
    """Profiled ML deviance (to minimise over theta).

    Evaluates:

        obj(theta) = log|A11| + n * log(y'Py)

    Differs from :func:`reml_objective` by omitting the ``log|X'Omega^{-1}X|``
    term and using ``n`` rather than ``n - p`` as the multiplier.

    Parameters
    ----------
    theta, y, X, Z, q_sizes, _cache, specs, n_levels:
        Same as :func:`reml_objective`.

    Returns
    -------
    float  (profiled ML deviance; lower is better)
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

    if specs is not None and not all(s.n_terms == 1 for s in specs):
        builder = _cache.get("lambda_builder") if _cache is not None else None
        if builder is not None:
            Lambda = builder.update(theta)  # type: ignore[union-attr]
        else:
            Lambda = make_lambda(theta, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
    else:
        _q_sizes = n_levels if specs is not None else q_sizes
        lambda_diag = make_lambda_diag(theta, _q_sizes)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty
        lZtX = lambda_diag[:, None] * ZtX

    chol_factor = _cache.get("chol_factor") if _cache is not None else None
    chol_api = _cache.get("chol_api", "old") if _cache is not None else "old"
    if chol_factor is not None:
        if chol_api == "new":
            chol_factor.factorize(A11)  # type: ignore[union-attr]
            log_det_A11 = float(chol_factor.logdet())  # type: ignore[union-attr]
            c1 = np.asarray(chol_factor.solve(lZty, "A")).squeeze()  # type: ignore[union-attr]
            C_X = np.asarray(chol_factor.solve(lZtX, "A"))  # type: ignore[union-attr]
        else:
            chol_factor.cholesky(A11)  # type: ignore[union-attr]
            log_det_A11 = float(chol_factor.logdet())  # type: ignore[union-attr]
            c1 = np.asarray(chol_factor.solve_A(lZty)).squeeze()  # type: ignore[union-attr]
            C_X = np.asarray(chol_factor.solve_A(lZtX))  # type: ignore[union-attr]
    else:
        log_det_A11 = sparse_chol_logdet(A11)
        c1 = _sparse_solve(A11, lZty)
        C_X = _sparse_solve(A11, lZtX)

    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1

    try:
        beta_hat = la.solve(MX, rhs, assume_a="pos")
    except la.LinAlgError:
        return np.inf

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    if yPy <= 0:
        return np.inf

    return float(log_det_A11 + n * np.log(yPy))


def fit_ml(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    theta0: np.ndarray | None = None,
    *,
    specs: list[RandomEffectSpec] | None = None,
    n_levels: list[int] | None = None,
    optimizer: str = "lbfgsb",
    weights: np.ndarray | None = None,
    correlation: CorStruct | None = None,
) -> REMLResult:
    """Fit a linear mixed model by profiled ML.

    Identical to :func:`fit_reml` but optimises the ML (not REML) criterion.
    The key differences are:

    * No ``log|X'Omega^{-1}X|`` correction term in the objective.
    * ``sigma2`` is estimated as ``y'Py / n`` (ML, biased) rather than
      ``y'Py / (n - p)`` (REML, unbiased).
    * The log-likelihood uses ``n`` degrees of freedom:
      ``llf = -0.5 * (log|A11| + n*(1 + log(2*pi*sigma2)))``.

    Parameters
    ----------
    y, X, Z, q_sizes, theta0, specs, n_levels, optimizer:
        Same as :func:`fit_reml`.
    weights:
        Observation-level prior weights, shape (n,). Defaults to ones.

    Returns
    -------
    REMLResult
        The ``llf`` field contains the ML log-likelihood.
    """
    if optimizer not in ("lbfgsb", "bobyqa", "nelder-mead"):
        msg = (
            f"optimizer must be 'lbfgsb', 'bobyqa', or 'nelder-mead', got {optimizer!r}"
        )
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

    # --- Correlation structure: joint optimisation over (theta, rho_raw) ---
    if correlation is not None:
        return _fit_ml_with_correlation(
            y,
            X,
            Z,
            q_sizes,
            theta0,
            specs=specs,
            n_levels=n_levels,
            optimizer=optimizer,
            weights=weights,
            correlation=correlation,
            n_theta=n_theta,
            bounds=bounds,
        )

    cache = _precompute(y, X, Z, weights=weights)

    if specs is not None and not all(s.n_terms == 1 for s in specs):
        cache["lambda_builder"] = LambdaBuilder(specs, n_levels)  # type: ignore[arg-type]

    cholmod = _try_cholmod()
    if cholmod is not None:
        if specs is not None and not all(s.n_terms == 1 for s in specs):
            Lambda0 = cache["lambda_builder"].update(theta0)  # type: ignore[union-attr]
            A11_0 = _build_A11(cache["ZtZ"], Lambda0)
        else:
            _q_init_ml = n_levels if specs is not None else q_sizes
            lambda_diag_0 = make_lambda_diag(theta0, _q_init_ml)  # type: ignore[arg-type]
            A11_0 = _build_A11(cache["ZtZ"], lambda_diag_0)
        factor, api = _init_chol_factor(cholmod, A11_0)
        if factor is not None:
            cache["chol_factor"] = factor
            cache["chol_api"] = api

    lower_bounds = np.array([lo if lo is not None else -np.inf for lo, _ in bounds])

    def obj(theta: np.ndarray) -> float:
        return ml_objective(
            theta, y, X, Z, q_sizes, _cache=cache, specs=specs, n_levels=n_levels
        )

    if optimizer == "bobyqa":
        import pybobyqa

        upper = np.array([hi if hi is not None else np.inf for _, hi in bounds])
        soln = pybobyqa.solve(obj, theta0, bounds=(lower_bounds, upper))
        theta_hat = soln.x
        converged = soln.msg == "Success: rho has reached rhoend"
    elif optimizer == "nelder-mead":

        def obj_bounded(theta: np.ndarray) -> float:
            return obj(np.maximum(theta, lower_bounds))

        res = opt.minimize(obj_bounded, theta0, method="Nelder-Mead")
        theta_hat = np.maximum(res.x, lower_bounds)
        converged = bool(res.success)
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

    if specs is not None and not all(s.n_terms == 1 for s in specs):
        Lambda = make_lambda(theta_hat, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
    else:
        _q_hat_ml = n_levels if specs is not None else q_sizes
        lambda_diag = make_lambda_diag(theta_hat, _q_hat_ml)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty
        lZtX = lambda_diag[:, None] * ZtX

    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1
    beta_hat = la.solve(MX, rhs, assume_a="pos")

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    sigma2 = yPy / n  # ML estimate (biased; uses n not n-p)

    # --- ML log-likelihood ---
    log_det_A11 = sparse_chol_logdet(A11)
    llf = -0.5 * (log_det_A11 + n * (1.0 + np.log(2.0 * np.pi * sigma2)))

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


# ---------------------------------------------------------------------------
# ML with residual correlation structure
# ---------------------------------------------------------------------------


def _fit_ml_with_correlation(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    q_sizes: list[int],
    theta0: np.ndarray,
    *,
    specs: list[RandomEffectSpec] | None = None,
    n_levels: list[int] | None = None,
    optimizer: str = "lbfgsb",
    weights: np.ndarray | None = None,
    correlation: CorStruct,
    n_theta: int,
    bounds: list[tuple[float | None, float | None]],
) -> REMLResult:
    """Fit ML with a residual correlation structure.

    Analogous to :func:`_fit_reml_with_correlation` but uses the ML criterion.
    """
    from interlace.correlation import rho_from_unconstrained, unconstrained_from_rho

    n, p = X.shape
    n_corr = correlation.n_corr_params

    rho_raw0 = np.array([unconstrained_from_rho(0.3)] * n_corr)

    if weights is not None:
        sqW = np.sqrt(weights)
        y_w0 = sqW * y
        X_w0 = sqW[:, None] * X
        Z_w0 = sp.diags(sqW, format="csc") @ Z
    else:
        y_w0 = y
        X_w0 = X
        Z_w0 = Z

    def obj_joint(params: np.ndarray) -> float:
        theta_re = params[:n_theta]
        rho_raw = params[n_theta:]
        rho = np.array([rho_from_unconstrained(float(r)) for r in rho_raw])

        y_w, X_w, Z_w = correlation.whiten_data(y_w0, X_w0, Z_w0, rho)
        cache_w = _precompute(y_w, X_w, Z_w)
        val = ml_objective(
            theta_re,
            y_w,
            X_w,
            Z_w,
            q_sizes,
            _cache=cache_w,
            specs=specs,
            n_levels=n_levels,
        )
        if not np.isfinite(val):
            return np.inf

        log_det_r = correlation.log_det_R(rho)
        return float(val + log_det_r)

    params0 = np.concatenate([theta0, rho_raw0])
    corr_bounds = correlation.unconstrained_bounds()
    bounds_joint = list(bounds) + corr_bounds
    lower_bounds_joint = np.array(
        [lo if lo is not None else -np.inf for lo, _ in bounds_joint]
    )

    if optimizer == "bobyqa":
        import pybobyqa

        upper = np.array([hi if hi is not None else np.inf for _, hi in bounds_joint])
        soln = pybobyqa.solve(obj_joint, params0, bounds=(lower_bounds_joint, upper))
        params_hat = soln.x
        converged = soln.msg == "Success: rho has reached rhoend"
    elif optimizer == "nelder-mead":

        def obj_bounded(params: np.ndarray) -> float:
            return obj_joint(np.maximum(params, lower_bounds_joint))

        res = opt.minimize(obj_bounded, params0, method="Nelder-Mead")
        params_hat = np.maximum(res.x, lower_bounds_joint)
        converged = bool(res.success)
    else:
        res = opt.minimize(
            obj_joint,
            params0,
            method="L-BFGS-B",
            bounds=bounds_joint,
        )
        params_hat = res.x
        converged = bool(res.success)

    theta_hat = params_hat[:n_theta]
    rho_raw_hat = params_hat[n_theta:]
    rho_hat = np.array([rho_from_unconstrained(float(r)) for r in rho_raw_hat])
    corr_params: dict[str, float] = {"rho": float(rho_hat[0])}

    # --- Recover beta and sigma2 at optimum ---
    y_w, X_w, Z_w = correlation.whiten_data(y_w0, X_w0, Z_w0, rho_hat)
    cache_w = _precompute(y_w, X_w, Z_w)

    ZtZ = sp.csc_matrix(cache_w["ZtZ"])
    ZtX = np.asarray(cache_w["ZtX"])
    Zty = np.asarray(cache_w["Zty"])
    XtX = np.asarray(cache_w["XtX"])
    Xty = np.asarray(cache_w["Xty"])
    yty = float(cache_w["yty"])

    if specs is not None and not all(s.n_terms == 1 for s in specs):
        Lambda = make_lambda(theta_hat, specs, n_levels)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, Lambda)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        lZtX = np.asarray(Lambda.T @ ZtX)
    else:
        _q_hat = n_levels if specs is not None else q_sizes
        lambda_diag = make_lambda_diag(theta_hat, _q_hat)  # type: ignore[arg-type]
        A11 = _build_A11(ZtZ, lambda_diag)
        lZty = lambda_diag * Zty
        lZtX = lambda_diag[:, None] * ZtX

    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X
    rhs = Xty - lZtX.T @ c1
    beta_hat = la.solve(MX, rhs, assume_a="pos")

    yPy = float(yty - lZty @ c1 - rhs @ beta_hat)
    sigma2 = yPy / n

    log_det_A11 = sparse_chol_logdet(A11)
    log_det_r = correlation.log_det_R(rho_hat)
    llf = -0.5 * (log_det_A11 + log_det_r + n * (1.0 + np.log(2.0 * np.pi * sigma2)))

    nparams = p + n_theta + n_corr + 1
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
        correlation_params=corr_params,
    )
