"""Satterthwaite denominator degrees of freedom for fixed-effect t-tests.

For fixed-effect coefficient j, the Satterthwaite approximated DF is:

    ν_j = 2 · V_jj² / [(∂V_jj/∂θ)' · H_D⁻¹ · (∂V_jj/∂θ)]

where:
- V_jj = fe_cov[j, j]  — estimated variance of β̂_j
- θ      — optimised REML variance parameters
- H_D    — Hessian of the REML deviance D(θ) = –2·llf(θ) w.r.t. θ
- Cov(θ) ≈ 2 · H_D⁻¹  (asymptotic covariance from REML information)

Both the gradient ∂V_jj/∂θ and H_D are computed by numerical central
finite differences.

Reference: Satterthwaite (1946); Kuznetsova, Brockhoff & Christensen (2017)
lmerTest, J. Stat. Softw. 82(13).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from interlace.profiled_reml import (
    _build_A11,
    _precompute,
    _sparse_solve,
    make_lambda,
    reml_objective,
)

if TYPE_CHECKING:
    from interlace.result import CrossedLMEResult


def satterthwaite_dfs(result: CrossedLMEResult) -> np.ndarray:
    """Compute Satterthwaite denominator DFs for all FE coefficients.

    Parameters
    ----------
    result:
        A fitted :class:`~interlace.result.CrossedLMEResult`.  Must have
        ``_Z`` (joint random-effects design matrix) and ``_n_levels``
        stored from the fitting step.

    Returns
    -------
    np.ndarray, shape (p,)
        Satterthwaite denominator degrees of freedom, one per FE coefficient.
        Entries are clipped to a minimum of 1 to avoid degenerate t-tests.
    """
    y: np.ndarray = result.model.endog
    X: np.ndarray = result.model.exog
    Z: Any = result._Z  # scipy sparse, shape (n, q)
    theta_hat: np.ndarray = result.theta
    specs = result._random_specs
    n_levels: list[int] = result._n_levels

    n, p = X.shape
    k = len(theta_hat)

    # Precompute cross-products once (reused across all finite-difference steps)
    cache = _precompute(y, X, Z)
    ZtX = np.asarray(cache["ZtX"])
    Zty = np.asarray(cache["Zty"])
    XtX = np.asarray(cache["XtX"])
    Xty = np.asarray(cache["Xty"])
    yty = float(cache["yty"])

    def _fe_cov_diag(theta: np.ndarray) -> np.ndarray:
        """Return diag(fe_cov) = σ²(θ) · diag((X'Ω⁻¹X)⁻¹) at given theta."""
        Lambda = make_lambda(theta, specs, n_levels)
        A11 = _build_A11(sp.csc_matrix(cache["ZtZ"]), Lambda)
        lZtX = np.asarray(Lambda.T @ ZtX)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        C_X = _sparse_solve(A11, lZtX)
        c1 = _sparse_solve(A11, lZty)
        MX = XtX - lZtX.T @ C_X
        rhs = Xty - lZtX.T @ c1
        try:
            beta = la.solve(MX, rhs, assume_a="pos")
        except la.LinAlgError:
            return np.full(p, np.nan)
        yPy = yty - lZty @ c1 - rhs @ beta
        if yPy <= 0:
            return np.full(p, np.nan)
        sigma2 = yPy / (n - p)
        MX_inv = np.linalg.inv(MX)
        return np.asarray(sigma2 * np.diag(MX_inv))

    # --- Step 1: gradient ∂V_jj/∂θ via central differences ---
    h_grad = 1e-4
    grad = np.zeros((k, p))  # grad[i, j] = ∂V_jj/∂θ_i
    for i in range(k):
        theta_p = theta_hat.copy()
        theta_m = theta_hat.copy()
        theta_p[i] += h_grad
        theta_m[i] -= h_grad
        vp = _fe_cov_diag(theta_p)
        vm = _fe_cov_diag(theta_m)
        grad[i, :] = (vp - vm) / (2.0 * h_grad)

    # --- Step 2: Hessian of REML deviance D(θ) via central differences ---
    def _deviance(theta: np.ndarray) -> float:
        return reml_objective(
            theta, y, X, Z, [], _cache=cache, specs=specs, n_levels=n_levels
        )

    h_hess = 1e-3
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h_hess
            ej[j] = h_hess
            H[i, j] = (
                _deviance(theta_hat + ei + ej)
                - _deviance(theta_hat + ei - ej)
                - _deviance(theta_hat - ei + ej)
                + _deviance(theta_hat - ei - ej)
            ) / (4.0 * h_hess**2)
            H[j, i] = H[i, j]

    # Regularise Hessian to ensure invertibility
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    # --- Step 3: Satterthwaite DF for each FE coefficient ---
    dfs = np.zeros(p)
    V_diag = np.diag(result.fe_cov)
    for j in range(p):
        g = grad[:, j]  # ∂V_jj/∂θ, shape (k,)
        V_jj = V_diag[j]
        denom = float(g @ H_inv @ g)
        if denom <= 0 or not np.isfinite(denom):
            dfs[j] = np.inf
        else:
            # Factor of 2: Cov(θ) ≈ 2·H_D⁻¹ → denominator = (∂g)'·2·H_D⁻¹·(∂g)
            # ν = 2·V_jj² / [(∂g)'·2·H_D⁻¹·(∂g)] = V_jj² / [(∂g)'·H_D⁻¹·(∂g)]
            dfs[j] = V_jj**2 / denom

    # Clip to minimum of 1 to keep t-distribution well-defined
    return np.maximum(dfs, 1.0)
