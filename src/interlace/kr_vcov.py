"""Kenward-Roger variance-covariance matrix derivatives.

Computes first and second derivatives of the fixed-effect variance-covariance
matrix C(theta) = sigma^2(theta) * (X' Omega^{-1}(theta) X)^{-1} with respect
to the REML variance parameters theta, via numerical central finite differences.

These are the core inputs for the Kenward-Roger denominator DF adjustment
(Kenward & Roger, 1997).  The downstream KR computation combines these
derivatives with the asymptotic covariance of theta-hat (Phi) to produce
bias-adjusted covariance and scaled F-statistics.

References
----------
Kenward, M.G. & Roger, J.H. (1997). Small sample inference for fixed effects
from restricted maximum likelihood. Biometrics, 53(3), 983-997.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class KRDerivatives:
    """Container for KR variance-covariance derivatives.

    Attributes
    ----------
    dC : np.ndarray, shape (k, p, p)
        First derivatives of fe_cov w.r.t. each theta_i.
    d2C : np.ndarray, shape (k, k, p, p)
        Second derivatives of fe_cov w.r.t. each (theta_i, theta_j) pair.
    Phi : np.ndarray, shape (k, k)
        Asymptotic covariance of theta-hat: ``2 * H_D^{-1}`` where H_D is
        the Hessian of the REML deviance w.r.t. theta.
    fe_cov : np.ndarray, shape (p, p)
        The fixed-effect covariance matrix at the REML optimum.
    """

    dC: np.ndarray
    d2C: np.ndarray
    Phi: np.ndarray
    fe_cov: np.ndarray


def kr_vcov_derivs(result: CrossedLMEResult) -> KRDerivatives:
    """Compute KR variance-covariance derivatives for a fitted LME model.

    Parameters
    ----------
    result :
        A fitted :class:`~interlace.result.CrossedLMEResult`.  Must have
        ``_Z`` (joint random-effects design matrix) and ``_n_levels``
        stored from the fitting step.

    Returns
    -------
    KRDerivatives
        First and second derivatives of the fixed-effect covariance matrix
        plus the asymptotic covariance of the variance parameters.
    """
    y: np.ndarray = result.model.endog
    X: np.ndarray = result.model.exog
    Z: Any = result._Z
    theta_hat: np.ndarray = result.theta
    specs = result._random_specs
    n_levels: list[int] = result._n_levels

    n, p = X.shape
    k = len(theta_hat)

    cache = _precompute(y, X, Z)
    ZtZ = sp.csc_matrix(cache["ZtZ"])
    ZtX = np.asarray(cache["ZtX"])
    Zty = np.asarray(cache["Zty"])
    XtX = np.asarray(cache["XtX"])
    Xty = np.asarray(cache["Xty"])
    yty = float(cache["yty"])

    def _fe_cov(theta: np.ndarray) -> np.ndarray:
        """Return the full p x p fe_cov matrix at given theta."""
        Lambda = make_lambda(theta, specs, n_levels)
        A11 = _build_A11(ZtZ, Lambda)
        lZtX = np.asarray(Lambda.T @ ZtX)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        C_X = _sparse_solve(A11, lZtX)
        c1 = _sparse_solve(A11, lZty)
        MX = XtX - lZtX.T @ C_X
        rhs = Xty - lZtX.T @ c1
        try:
            beta = la.solve(MX, rhs, assume_a="pos")
        except la.LinAlgError:
            return np.full((p, p), np.nan)
        yPy = yty - lZty @ c1 - rhs @ beta
        if yPy <= 0:
            return np.full((p, p), np.nan)
        sigma2 = yPy / (n - p)
        MX_inv = np.linalg.inv(MX)
        return np.asarray(sigma2 * MX_inv)

    # --- Step 1: first derivatives dC/dtheta_i via central differences ---
    h_grad = 1e-4
    dC = np.zeros((k, p, p))
    for i in range(k):
        theta_p = theta_hat.copy()
        theta_m = theta_hat.copy()
        theta_p[i] += h_grad
        theta_m[i] -= h_grad
        dC[i] = (_fe_cov(theta_p) - _fe_cov(theta_m)) / (2.0 * h_grad)

    # --- Step 2: second derivatives d2C/dtheta_i dtheta_j via 4-point formula ---
    h_hess = 1e-3
    d2C = np.zeros((k, k, p, p))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h_hess
            ej[j] = h_hess
            C_pp = _fe_cov(theta_hat + ei + ej)
            C_pm = _fe_cov(theta_hat + ei - ej)
            C_mp = _fe_cov(theta_hat - ei + ej)
            C_mm = _fe_cov(theta_hat - ei - ej)
            d2C[i, j] = (C_pp - C_pm - C_mp + C_mm) / (4.0 * h_hess**2)
            if j != i:
                d2C[j, i] = d2C[i, j]

    # --- Step 3: REML Hessian and Phi = 2 * H_D^{-1} ---
    def _deviance(theta: np.ndarray) -> float:
        return reml_objective(
            theta, y, X, Z, [], _cache=cache, specs=specs, n_levels=n_levels
        )

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

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    Phi = 2.0 * H_inv

    return KRDerivatives(
        dC=dC,
        d2C=d2C,
        Phi=Phi,
        fe_cov=np.asarray(result.fe_cov),
    )
