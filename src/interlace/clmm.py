"""Cumulative link mixed model (CLMM) — proportional odds with random effects.

Implements the model used by R's ``ordinal::clmm()``:

    link(P(Y <= k | b)) = alpha_k - x'beta - z'b,   k = 1,...,K-1

where alpha_1 < alpha_2 < ... < alpha_{K-1} are threshold parameters,
beta are fixed-effect coefficients (no intercept — absorbed into thresholds),
and b ~ N(0, sigma^2 * Lambda Lambda') are random effects.

The marginal likelihood is approximated via Laplace (PIRLS inner loop +
outer optimisation over variance parameters).

References
----------
Christensen, R.H.B. (2019). ordinal — Regression Models for Ordinal Data.
    R package version 2019.12-10.
Agresti, A. (2010). Analysis of Ordinal Categorical Data, 2nd ed. Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from interlace.formula import groups_to_random_effects, parse_random_effects
from interlace.profiled_reml import (
    _build_theta_bounds,
    make_lambda,
    n_theta_for_spec,
    sparse_chol_logdet,
)
from interlace.sparse_z import build_joint_z_from_specs, group_array

# ---------------------------------------------------------------------------
# Link functions: CDF, PDF, PDF derivative
# ---------------------------------------------------------------------------

_EXP_MAX = 500.0  # conservative clamp for exp()
_PROB_EPS = 1e-12  # floor for category probabilities


def _logit_cdf(x: np.ndarray) -> np.ndarray:
    """Logistic CDF: 1 / (1 + exp(-x))."""
    from scipy.special import expit

    return np.asarray(expit(x), dtype=np.float64)


def _logit_pdf(x: np.ndarray) -> np.ndarray:
    """Logistic PDF: F(x) * (1 - F(x))."""
    F = _logit_cdf(x)
    return F * (1.0 - F)


def _logit_pdf_deriv(x: np.ndarray) -> np.ndarray:
    """Logistic PDF derivative: f(x) * (1 - 2*F(x))."""
    F = _logit_cdf(x)
    f = F * (1.0 - F)
    return f * (1.0 - 2.0 * F)


def _probit_cdf(x: np.ndarray) -> np.ndarray:
    """Normal CDF."""
    from scipy.special import ndtr

    return np.asarray(ndtr(x), dtype=np.float64)


def _probit_pdf(x: np.ndarray) -> np.ndarray:
    """Normal PDF."""
    return np.asarray(np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi))


def _probit_pdf_deriv(x: np.ndarray) -> np.ndarray:
    """Normal PDF derivative: -x * phi(x)."""
    return np.asarray(-x * _probit_pdf(x))


def _cloglog_cdf(x: np.ndarray) -> np.ndarray:
    """Complementary log-log CDF: 1 - exp(-exp(x))."""
    ex = np.exp(np.clip(x, -_EXP_MAX, _EXP_MAX))
    return np.asarray(1.0 - np.exp(-ex))


def _cloglog_pdf(x: np.ndarray) -> np.ndarray:
    """Complementary log-log PDF: exp(x - exp(x))."""
    xc = np.clip(x, -_EXP_MAX, _EXP_MAX)
    return np.asarray(np.exp(xc - np.exp(xc)))


def _cloglog_pdf_deriv(x: np.ndarray) -> np.ndarray:
    """Complementary log-log PDF derivative: (1 - exp(x)) * exp(x - exp(x))."""
    xc = np.clip(x, -_EXP_MAX, _EXP_MAX)
    ex = np.exp(xc)
    return np.asarray((1.0 - ex) * np.exp(xc - ex))


_LINKS = {
    "logit": (_logit_cdf, _logit_pdf, _logit_pdf_deriv),
    "probit": (_probit_cdf, _probit_pdf, _probit_pdf_deriv),
    "cloglog": (_cloglog_cdf, _cloglog_pdf, _cloglog_pdf_deriv),
}


# ---------------------------------------------------------------------------
# Threshold parameterisation
# ---------------------------------------------------------------------------


def _increments_to_thresholds(alpha1: float, log_deltas: np.ndarray) -> np.ndarray:
    """Convert (alpha_1, log_delta_2, ..., log_delta_{K-1}) to thresholds.

    alpha_k = alpha_1 + sum_{j=2}^{k} exp(log_delta_j)

    This ensures strict ordering: alpha_1 < alpha_2 < ... < alpha_{K-1}.
    """
    K_minus_1 = 1 + len(log_deltas)
    thresholds = np.empty(K_minus_1)
    thresholds[0] = alpha1
    cum = alpha1
    for j in range(len(log_deltas)):
        cum += np.exp(log_deltas[j])
        thresholds[j + 1] = cum
    return thresholds


def _thresholds_to_increments(thresholds: np.ndarray) -> tuple[float, np.ndarray]:
    """Inverse of _increments_to_thresholds: extract alpha_1 and log-deltas."""
    alpha1 = float(thresholds[0])
    if len(thresholds) == 1:
        return alpha1, np.array([])
    diffs = np.diff(thresholds)
    log_deltas = np.log(np.maximum(diffs, 1e-10))
    return alpha1, log_deltas


# ---------------------------------------------------------------------------
# Ordinal log-likelihood and PIRLS working quantities
# ---------------------------------------------------------------------------


def _category_probs(
    thresholds: np.ndarray,
    eta: np.ndarray,
    cdf_fn: Any,
) -> np.ndarray:
    """Compute category probabilities P(Y=k) for all observations.

    Parameters
    ----------
    thresholds : shape (K-1,), the alpha_k values.
    eta : shape (n,), linear predictor x'beta + z'b.
    cdf_fn : CDF function F.

    Returns
    -------
    probs : shape (n, K), where probs[i, k] = P(Y_i = k+1).
    """
    n = len(eta)
    K = len(thresholds) + 1
    probs = np.empty((n, K))

    # Cumulative probs: gamma_k = alpha_k - eta
    # P(Y <= k) = F(gamma_k)
    # P(Y = k) = F(gamma_k) - F(gamma_{k-1})
    # Convention: F(gamma_0) = 0, F(gamma_K) = 1

    for k in range(K):
        if k == 0:
            F_upper = cdf_fn(thresholds[0] - eta)
            probs[:, 0] = F_upper
        elif k == K - 1:
            F_lower = cdf_fn(thresholds[k - 1] - eta)
            probs[:, k] = 1.0 - F_lower
        else:
            F_upper = cdf_fn(thresholds[k] - eta)
            F_lower = cdf_fn(thresholds[k - 1] - eta)
            probs[:, k] = F_upper - F_lower

    # Clamp to avoid log(0)
    np.clip(probs, _PROB_EPS, 1.0, out=probs)
    return probs


def _ordinal_loglik(
    y_codes: np.ndarray,
    thresholds: np.ndarray,
    eta: np.ndarray,
    cdf_fn: Any,
) -> float:
    """Conditional log-likelihood for ordinal data.

    Parameters
    ----------
    y_codes : shape (n,), integer codes 0, 1, ..., K-1.
    thresholds : shape (K-1,).
    eta : shape (n,), linear predictor.
    cdf_fn : CDF function.

    Returns
    -------
    Total log-likelihood (scalar).
    """
    probs = _category_probs(thresholds, eta, cdf_fn)
    n = len(y_codes)
    ll = 0.0
    for i in range(n):
        ll += np.log(probs[i, y_codes[i]])
    return ll


def _ordinal_score_hessian(
    y_codes: np.ndarray,
    thresholds: np.ndarray,
    eta: np.ndarray,
    cdf_fn: Any,
    pdf_fn: Any,
    pdf_deriv_fn: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Score and negative Hessian of ordinal log-likelihood w.r.t. eta.

    For observation i with Y_i = k (0-indexed):
        p_k = F(alpha_k - eta) - F(alpha_{k-1} - eta)
        score_i = d(log p_k)/d(eta) = -(f_upper - f_lower) / p_k
        neg_hess_i = -d^2(log p_k)/d(eta)^2

    where f_upper = f(alpha_k - eta), f_lower = f(alpha_{k-1} - eta),
    and f_upper = 0 for k=K-1 (last category), f_lower = 0 for k=0.

    Returns
    -------
    score : shape (n,)
    neg_hess : shape (n,), always positive (floored at 1e-10).
    """
    n = len(y_codes)
    K = len(thresholds) + 1
    score = np.empty(n)
    neg_hess = np.empty(n)

    for i in range(n):
        k = y_codes[i]

        # Upper boundary: alpha_k - eta (for k < K-1)
        # Lower boundary: alpha_{k-1} - eta (for k > 0)
        if k == 0:
            gamma_u = thresholds[0] - eta[i]
            F_u = float(cdf_fn(np.array([gamma_u]))[0])
            f_u = float(pdf_fn(np.array([gamma_u]))[0])
            fp_u = float(pdf_deriv_fn(np.array([gamma_u]))[0])
            F_l = 0.0
            f_l = 0.0
            fp_l = 0.0
        elif k == K - 1:
            gamma_l = thresholds[k - 1] - eta[i]
            F_u = 1.0
            f_u = 0.0
            fp_u = 0.0
            F_l = float(cdf_fn(np.array([gamma_l]))[0])
            f_l = float(pdf_fn(np.array([gamma_l]))[0])
            fp_l = float(pdf_deriv_fn(np.array([gamma_l]))[0])
        else:
            gamma_u = thresholds[k] - eta[i]
            gamma_l = thresholds[k - 1] - eta[i]
            F_u = float(cdf_fn(np.array([gamma_u]))[0])
            f_u = float(pdf_fn(np.array([gamma_u]))[0])
            fp_u = float(pdf_deriv_fn(np.array([gamma_u]))[0])
            F_l = float(cdf_fn(np.array([gamma_l]))[0])
            f_l = float(pdf_fn(np.array([gamma_l]))[0])
            fp_l = float(pdf_deriv_fn(np.array([gamma_l]))[0])

        p_k = max(F_u - F_l, _PROB_EPS)

        # Score: d(log p_k)/d(eta) = -(f_u - f_l) / p_k
        # (negative sign because gamma = alpha - eta, so d(gamma)/d(eta) = -1)
        s_i = -(f_u - f_l) / p_k

        # Negative Hessian: -d^2(log p_k)/d(eta)^2
        # d^2(log p_k)/d(eta)^2 = (f'_u - f'_l)/p_k - s^2
        # so neg_hess = (f'_l - f'_u)/p_k + s^2
        nh_i = (fp_l - fp_u) / p_k + s_i**2

        score[i] = s_i
        neg_hess[i] = nh_i

    # Floor negative Hessian to ensure positive working weights
    np.maximum(neg_hess, 1e-10, out=neg_hess)
    return score, neg_hess


def _ordinal_score_hessian_vec(
    y_codes: np.ndarray,
    thresholds: np.ndarray,
    eta: np.ndarray,
    cdf_fn: Any,
    pdf_fn: Any,
    pdf_deriv_fn: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised score and negative Hessian (same semantics, faster)."""
    n = len(y_codes)
    K = len(thresholds) + 1

    # For each observation, determine upper/lower gamma
    # y_codes[i] = k means category k (0-indexed)
    # gamma_upper = thresholds[k] - eta  (if k < K-1, else inf)
    # gamma_lower = thresholds[k-1] - eta  (if k > 0, else -inf)

    gamma_upper = np.full(n, np.inf)
    gamma_lower = np.full(n, -np.inf)

    not_last = y_codes < K - 1
    not_first = y_codes > 0

    if np.any(not_last):
        gamma_upper[not_last] = thresholds[y_codes[not_last]] - eta[not_last]
    if np.any(not_first):
        gamma_lower[not_first] = thresholds[y_codes[not_first] - 1] - eta[not_first]

    # CDF values
    F_u = np.ones(n)  # F(inf) = 1
    F_l = np.zeros(n)  # F(-inf) = 0
    f_u = np.zeros(n)
    f_l = np.zeros(n)
    fp_u = np.zeros(n)
    fp_l = np.zeros(n)

    if np.any(not_last):
        gu = gamma_upper[not_last]
        F_u[not_last] = cdf_fn(gu)
        f_u[not_last] = pdf_fn(gu)
        fp_u[not_last] = pdf_deriv_fn(gu)
    if np.any(not_first):
        gl = gamma_lower[not_first]
        F_l[not_first] = cdf_fn(gl)
        f_l[not_first] = pdf_fn(gl)
        fp_l[not_first] = pdf_deriv_fn(gl)

    p_k = np.maximum(F_u - F_l, _PROB_EPS)

    score = -(f_u - f_l) / p_k
    neg_hess = (fp_l - fp_u) / p_k + score**2
    np.maximum(neg_hess, 1e-10, out=neg_hess)

    return score, neg_hess


# ---------------------------------------------------------------------------
# Score w.r.t. thresholds (for gradient of outer objective)
# ---------------------------------------------------------------------------


def _threshold_score(
    y_codes: np.ndarray,
    thresholds: np.ndarray,
    eta: np.ndarray,
    cdf_fn: Any,
    pdf_fn: Any,
) -> np.ndarray:
    """Score of ordinal log-likelihood w.r.t. each threshold alpha_j.

    d(log p_{y_i})/d(alpha_j) is non-zero only when y_i == j or y_i == j+1
    (0-indexed: alpha_j is the j-th threshold, separating categories j and j+1).

    Returns
    -------
    grad_alpha : shape (K-1,), gradient w.r.t. each threshold.
    """
    K_minus_1 = len(thresholds)
    K = K_minus_1 + 1
    grad = np.zeros(K_minus_1)

    # Precompute category probs and PDF values at all boundaries
    # For threshold j (0-indexed), gamma_j = alpha_j - eta
    # f(gamma_j) = pdf_fn(gamma_j)

    for j in range(K_minus_1):
        gamma_j = thresholds[j] - eta
        f_j = pdf_fn(gamma_j)

        # Observations in category j (upper boundary is alpha_j):
        # d(log p_j)/d(alpha_j) = f_j / p_j
        mask_upper = y_codes == j
        if np.any(mask_upper):
            F_u = cdf_fn(gamma_j[mask_upper])
            if j == 0:
                F_l = np.zeros(np.sum(mask_upper))
            else:
                F_l = cdf_fn(thresholds[j - 1] - eta[mask_upper])
            p_k = np.maximum(F_u - F_l, _PROB_EPS)
            grad[j] += np.sum(f_j[mask_upper] / p_k)

        # Observations in category j+1 (lower boundary is alpha_j):
        # d(log p_{j+1})/d(alpha_j) = -f_j / p_{j+1}
        mask_lower = y_codes == j + 1
        if np.any(mask_lower):
            if j + 1 == K - 1:
                F_u2 = np.ones(np.sum(mask_lower))
            else:
                F_u2 = cdf_fn(thresholds[j + 1] - eta[mask_lower])
            F_l2 = cdf_fn(gamma_j[mask_lower])
            p_k2 = np.maximum(F_u2 - F_l2, _PROB_EPS)
            grad[j] -= np.sum(f_j[mask_lower] / p_k2)

    return grad


# ---------------------------------------------------------------------------
# PIRLS inner loop for CLMM
# ---------------------------------------------------------------------------

_PIRLS_MAXITER = 100
_PIRLS_TOL = 1e-8


def _clmm_pirls(
    y_codes: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    thresholds: np.ndarray,
    theta: np.ndarray,
    specs: list[Any],
    n_levels: list[int],
    cdf_fn: Any,
    pdf_fn: Any,
    pdf_deriv_fn: Any,
    u0: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """PIRLS for the CLMM.

    Solves for (beta, u) given fixed thresholds and theta.

    Parameters
    ----------
    y_codes : Integer category codes, shape (n,).
    X : Fixed effects design (no intercept), shape (n, p).
    Z : Random effects design, shape (n, q).
    thresholds : Current threshold values, shape (K-1,).
    theta : Variance parameters.
    specs : Random effect specifications.
    n_levels : Number of levels per grouping factor.
    cdf_fn, pdf_fn, pdf_deriv_fn : Link function components.
    u0, beta0 : Warm-start values.

    Returns
    -------
    beta, u, eta, laplace_ll, converged
    """
    n, p = X.shape
    q = Z.shape[1]

    Lambda = make_lambda(theta, specs, n_levels)

    u = np.zeros(q) if u0 is None else u0.copy()
    beta = np.zeros(p) if beta0 is None else beta0.copy()

    converged = False
    v_new = np.zeros(q)

    for _iteration in range(_PIRLS_MAXITER):
        eta = X @ beta + Z @ u

        # Working quantities from ordinal likelihood
        score, neg_hess = _ordinal_score_hessian_vec(
            y_codes, thresholds, eta, cdf_fn, pdf_fn, pdf_deriv_fn
        )

        w = neg_hess  # working weights
        z_w = eta + score / neg_hess  # working response

        # Standard PIRLS solve (same structure as glmm_laplace.py)
        sqrtW = np.sqrt(w)
        WX = sqrtW[:, None] * X
        Wz = sqrtW * z_w

        Z_star = Z @ Lambda
        WZs = sp.diags(sqrtW, format="csc") @ Z_star

        XtWX = WX.T @ WX
        ZstWX = np.asarray(
            (WZs.T @ WX).toarray() if sp.issparse(WZs.T @ WX) else WZs.T @ WX
        )
        ZstWZs = (WZs.T @ WZs).tocsc()
        XtWz = WX.T @ Wz
        ZstWz = np.asarray(WZs.T @ Wz).squeeze()

        A = (ZstWZs + sp.eye(q, format="csc")).tocsc()

        # Schur complement solve
        A_inv_ZstWX = np.column_stack([spla.spsolve(A, ZstWX[:, j]) for j in range(p)])
        A_inv_ZstWz = spla.spsolve(A, ZstWz)

        schur = XtWX - ZstWX.T @ A_inv_ZstWX
        rhs_beta = XtWz - ZstWX.T @ A_inv_ZstWz

        try:
            beta_new = la.solve(schur, rhs_beta, assume_a="pos")
        except la.LinAlgError:
            beta_new = la.lstsq(schur, rhs_beta)[0]

        v_new = A_inv_ZstWz - A_inv_ZstWX @ beta_new
        u_new = np.asarray(Lambda @ v_new).squeeze()

        # Step-halving for stability
        max_step = 5.0
        delta_beta_raw = beta_new - beta
        delta_u_raw = u_new - u
        max_delta = max(
            np.max(np.abs(delta_beta_raw)) if p > 0 else 0.0,
            np.max(np.abs(delta_u_raw)) if q > 0 else 0.0,
        )
        if max_delta > max_step:
            scale = max_step / max_delta
            beta_new = beta + scale * delta_beta_raw
            u_new = u + scale * delta_u_raw

        delta = max(
            np.max(np.abs(beta_new - beta)) if p > 0 else 0.0,
            np.max(np.abs(u_new - u)) if q > 0 else 0.0,
        )
        beta = beta_new
        u = u_new

        if delta < _PIRLS_TOL:
            converged = True
            break

    # Final linear predictor
    eta = X @ beta + Z @ u

    # Penalty
    penalty = float(v_new @ v_new)

    # Recompute A at final values for log|A|
    score_f, neg_hess_f = _ordinal_score_hessian_vec(
        y_codes, thresholds, eta, cdf_fn, pdf_fn, pdf_deriv_fn
    )
    w_f = neg_hess_f
    WZs_f = sp.diags(np.sqrt(w_f), format="csc") @ Z_star
    A_f = ((WZs_f.T @ WZs_f) + sp.eye(q, format="csc")).tocsc()
    log_det_A = sparse_chol_logdet(A_f)

    # Conditional log-likelihood
    cond_ll = _ordinal_loglik(y_codes, thresholds, eta, cdf_fn)

    # Laplace: ll = cond_ll - 0.5*penalty - 0.5*log|A|
    laplace_ll = cond_ll - 0.5 * penalty - 0.5 * log_det_A

    return beta, u, eta, laplace_ll, converged


# ---------------------------------------------------------------------------
# Outer objective: optimise over (thresholds, theta)
# ---------------------------------------------------------------------------


def _clmm_objective(
    params: np.ndarray,
    n_thresh_params: int,
    n_beta: int,
    y_codes: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    specs: list[Any],
    n_levels: list[int],
    cdf_fn: Any,
    pdf_fn: Any,
    pdf_deriv_fn: Any,
    warm: dict[str, Any],
) -> float:
    """Negative Laplace log-likelihood over (threshold_params, theta).

    params layout: [alpha_1, log_delta_2, ..., log_delta_{K-1}, theta_1, ...]

    Beta is profiled out via PIRLS (solved internally).
    """
    alpha1 = params[0]
    log_deltas = params[1:n_thresh_params]
    theta = params[n_thresh_params:]

    thresholds = _increments_to_thresholds(alpha1, log_deltas)

    beta, u, _eta, ll, _conv = _clmm_pirls(
        y_codes,
        X,
        Z,
        thresholds,
        theta,
        specs,
        n_levels,
        cdf_fn,
        pdf_fn,
        pdf_deriv_fn,
        u0=warm.get("u"),
        beta0=warm.get("beta"),
    )
    warm["u"] = u
    warm["beta"] = beta

    if not np.isfinite(ll):
        return 1e20
    return -ll


# ---------------------------------------------------------------------------
# Joint objective: optimise (threshold_params, beta, theta) together
# ---------------------------------------------------------------------------


def _clmm_joint_objective(
    params: np.ndarray,
    n_thresh_params: int,
    n_beta: int,
    y_codes: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    specs: list[Any],
    n_levels: list[int],
    cdf_fn: Any,
    pdf_fn: Any,
    pdf_deriv_fn: Any,
    warm: dict[str, Any],
) -> float:
    """Joint objective over (threshold_params, beta, theta).

    In this variant beta is taken from the outer optimiser (not profiled
    via PIRLS). PIRLS only solves for u.
    """
    alpha1 = params[0]
    log_deltas = params[1:n_thresh_params]
    beta_fixed = params[n_thresh_params : n_thresh_params + n_beta]
    theta = params[n_thresh_params + n_beta :]

    thresholds = _increments_to_thresholds(alpha1, log_deltas)

    q = Z.shape[1]
    Lambda = make_lambda(theta, specs, n_levels)
    Z_star = Z @ Lambda

    # PIRLS for u only, with beta fixed
    u_cached = warm.get("u")
    u = np.zeros(q) if u_cached is None else u_cached.copy()

    for _iteration in range(_PIRLS_MAXITER):
        eta = X @ beta_fixed + Z @ u

        score, neg_hess = _ordinal_score_hessian_vec(
            y_codes, thresholds, eta, cdf_fn, pdf_fn, pdf_deriv_fn
        )
        w = neg_hess

        # Solve for v only: (Z_*'WZ_* + I) v = Z_*'W * (z_w - X@beta)
        z_w = eta + score / neg_hess
        residual = z_w - X @ beta_fixed
        sqrtW = np.sqrt(w)
        WZs = sp.diags(sqrtW, format="csc") @ Z_star
        ZstWr = np.asarray(Z_star.T @ (w * residual)).squeeze()
        A = ((WZs.T @ WZs) + sp.eye(q, format="csc")).tocsc()
        v_new = spla.spsolve(A, ZstWr)
        u_new = np.asarray(Lambda @ v_new).squeeze()

        delta_u = np.max(np.abs(u_new - u))
        u = u_new

        if delta_u < _PIRLS_TOL:
            break

    warm["u"] = u

    # Compute Laplace log-likelihood
    eta = X @ beta_fixed + Z @ u
    cond_ll = _ordinal_loglik(y_codes, thresholds, eta, cdf_fn)
    penalty = float(v_new @ v_new)

    score_f, neg_hess_f = _ordinal_score_hessian_vec(
        y_codes, thresholds, eta, cdf_fn, pdf_fn, pdf_deriv_fn
    )
    WZs_f = sp.diags(np.sqrt(neg_hess_f), format="csc") @ Z_star
    A_f = ((WZs_f.T @ WZs_f) + sp.eye(q, format="csc")).tocsc()
    log_det_A = sparse_chol_logdet(A_f)

    ll = cond_ll - 0.5 * penalty - 0.5 * log_det_A

    if not np.isfinite(ll):
        return 1e20
    return -ll


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CLMMResult:
    """Result container for a fitted CLMM.

    Attributes
    ----------
    thresholds : dict
        Threshold estimates, keyed by label (e.g. ``"1|2"``, ``"2|3"``).
    threshold_bse : dict
        Standard errors of threshold estimates.
    fe_params : pd.Series
        Fixed-effect coefficient estimates.
    fe_bse : pd.Series
        Standard errors of fixed effects.
    random_effects : dict
        BLUPs per grouping factor.
    variance_components : dict
        Estimated variance for each grouping factor.
    theta : np.ndarray
        Raw variance parameters.
    converged : bool
        Whether the optimiser converged.
    nobs : int
        Number of observations.
    llf : float
        Laplace-approximated log-likelihood.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    ngroups : dict
        Number of levels per grouping factor.
    link : str
        Link function name.
    """

    thresholds: dict[str, float]
    threshold_bse: dict[str, float]
    fe_params: pd.Series
    fe_bse: pd.Series
    random_effects: dict[str, Any]
    variance_components: dict[str, float]
    theta: np.ndarray
    converged: bool
    nobs: int
    llf: float
    aic: float
    bic: float
    ngroups: dict[str, int]
    link: str
    _formula: str = ""
    _group_cols: list[str] | None = None

    def summary(self) -> _CLMMSummary:
        return _CLMMSummary(self)

    def predict(
        self,
        newdata: Any,
        type: str = "prob",
    ) -> np.ndarray:
        """Predict from the fitted CLMM.

        Parameters
        ----------
        newdata :
            DataFrame with the same covariates (and group columns) as
            the training data.
        type :
            ``"prob"`` — category probabilities P(Y=k), shape (n, K).
            ``"cum.prob"`` — cumulative probabilities P(Y<=k), shape (n, K-1).
            ``"linear.predictor"`` — eta = X'beta + Z'b, shape (n,).

        Returns
        -------
        np.ndarray
        """
        valid_types = ("prob", "cum.prob", "linear.predictor")
        if type not in valid_types:
            raise ValueError(f"Unknown type '{type}'. Choose from: {valid_types}")

        import formulaic

        cdf_fn, _, _ = _LINKS[self.link]
        thresholds_arr = np.array(list(self.thresholds.values()))

        # Build X from formula (same processing as fit_clmm)
        _, rhs = self._formula.split("~", 1)
        mm_full = formulaic.model_matrix("~ " + rhs.strip(), newdata)
        full_cols = list(mm_full.columns)
        if "Intercept" in full_cols:
            keep = [c for c in full_cols if c != "Intercept"]
            mm = mm_full[keep]
        else:
            mm = mm_full
        X = np.asarray(mm, dtype=np.float64)

        # Fixed-effects linear predictor
        eta: np.ndarray = np.asarray(X @ np.asarray(self.fe_params), dtype=np.float64)

        # Add random effects (BLUPs) for known groups; unseen -> 0
        group_cols = self._group_cols or []
        for grp in group_cols:
            if grp not in self.random_effects:
                continue
            re = self.random_effects[grp]
            col_vals = np.asarray(newdata[grp])
            for i, val in enumerate(col_vals):
                if val in re.index:
                    eta[i] += re.loc[val]

        if type == "linear.predictor":
            return eta

        # Cumulative probabilities: P(Y <= k) = F(alpha_k - eta)
        n = len(eta)
        K_minus_1 = len(thresholds_arr)
        cum_probs = np.empty((n, K_minus_1))
        for k in range(K_minus_1):
            cum_probs[:, k] = cdf_fn(thresholds_arr[k] - eta)

        if type == "cum.prob":
            return cum_probs

        # Category probabilities: P(Y = k)
        K = K_minus_1 + 1
        probs = np.empty((n, K))
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, K_minus_1):
            probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]
        probs[:, K - 1] = 1.0 - cum_probs[:, K_minus_1 - 1]
        np.clip(probs, 0.0, 1.0, out=probs)
        return probs

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        """Wald confidence intervals for thresholds and fixed effects.

        Parameters
        ----------
        level :
            Confidence level (default 0.95).

        Returns
        -------
        pd.DataFrame with columns ``["lower", "upper"]``.
        """
        from scipy.stats import norm

        z = norm.ppf((1.0 + level) / 2.0)

        names: list[str] = []
        estimates: list[float] = []
        ses: list[float] = []

        for name, val in self.thresholds.items():
            names.append(name)
            estimates.append(val)
            ses.append(self.threshold_bse[name])

        for name, val in zip(self.fe_params.index, self.fe_params.values, strict=True):
            names.append(str(name))
            estimates.append(float(val))
            ses.append(float(self.fe_bse.loc[name]))

        est = np.array(estimates)
        se = np.array(ses)
        lower = est - z * se
        upper = est + z * se

        return pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=names,
        )


class _CLMMSummary:
    def __init__(self, result: CLMMResult) -> None:
        self._result = result

    def __str__(self) -> str:
        return self._render()

    def __repr__(self) -> str:
        return self._render()

    def _render(self) -> str:
        r = self._result
        lines: list[str] = []
        lines.append("Cumulative link mixed model fit by Laplace")
        lines.append(f"Link: {r.link}")
        lines.append(f"Formula: {r._formula}")
        lines.append("")

        lines.append("Threshold coefficients:")
        header = f"  {'':12} {'Estimate':>12} {'Std. Error':>12}"
        lines.append(header)
        for name in r.thresholds:
            est = r.thresholds[name]
            se = r.threshold_bse.get(name, float("nan"))
            lines.append(f"  {name:<12} {est:>12.4f} {se:>12.4f}")
        lines.append("")

        lines.append("Fixed effects:")
        fe_arr = np.asarray(r.fe_params)
        bse_arr = np.asarray(r.fe_bse)
        z_arr = fe_arr / bse_arr
        names = list(r.fe_params.index)
        header2 = f"  {'':20} {'Estimate':>12} {'Std. Error':>12} {'z value':>10}"
        lines.append(header2)
        for name, est, se, zv in zip(names, fe_arr, bse_arr, z_arr, strict=True):
            lines.append(f"  {name:<20} {est:>12.4f} {se:>12.4f} {zv:>10.4f}")
        lines.append("")

        lines.append("Random effects:")
        for grp, vc in r.variance_components.items():
            n_grp = r.ngroups[grp]
            sd = np.sqrt(vc)
            lines.append(f"  {grp:<15} Var: {vc:.6f}  SD: {sd:.6f}  (n={n_grp})")
        lines.append("")

        groups_str = "; ".join(f"{g}: {n}" for g, n in r.ngroups.items())
        lines.append(f"Number of obs: {r.nobs}, groups: {groups_str}")
        lines.append(f"AIC: {r.aic:.2f}  BIC: {r.bic:.2f}  logLik: {r.llf:.2f}")
        status = "converged" if r.converged else "DID NOT CONVERGE"
        lines.append(f"Optimizer: {status}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_clmm(
    formula: str,
    data: Any,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    link: str = "logit",
    optimizer: str = "lbfgsb",
    theta0: np.ndarray | None = None,
) -> CLMMResult:
    """Fit a cumulative link mixed model (proportional odds).

    Parameters
    ----------
    formula :
        Fixed-effects formula, e.g. ``"rating ~ temp + contact"``.
        The intercept is suppressed (thresholds serve as intercepts).
    data :
        DataFrame (pandas, polars, or narwhals-compatible).
    groups :
        Column name(s) for random intercepts.
    random :
        lme4-style random effect specs (precedence over ``groups``).
    link :
        Link function: ``"logit"`` (default), ``"probit"``, ``"cloglog"``.
    optimizer :
        ``"lbfgsb"`` (default) or ``"bobyqa"``.
    theta0 :
        Initial variance parameters. Defaults to ones.

    Returns
    -------
    CLMMResult
    """
    import formulaic
    import narwhals as nw

    if link not in _LINKS:
        raise ValueError(f"Unknown link '{link}'. Choose from: {sorted(_LINKS)}")
    cdf_fn, pdf_fn, pdf_deriv_fn = _LINKS[link]

    nw_data = nw.from_native(data, eager_only=True)

    # --- Random effect specs ---
    if random is not None:
        specs = parse_random_effects(random)
    elif groups is not None:
        specs = groups_to_random_effects(groups)
    else:
        raise ValueError("Either 'groups' or 'random' must be provided.")

    # --- Parse ordinal response ---
    lhs, rhs = formula.split("~", 1)
    y_col = lhs.strip()
    y_raw = nw_data[y_col].to_numpy()

    # Convert to integer codes (0-indexed) and determine category labels
    unique_sorted = np.sort(np.unique(y_raw))
    K = len(unique_sorted)
    if K < 2:
        raise ValueError("Response must have at least 2 categories.")

    code_map = {v: i for i, v in enumerate(unique_sorted)}
    y_codes = np.array([code_map[v] for v in y_raw], dtype=np.intp)

    # Threshold labels: "1|2", "2|3", etc.
    threshold_labels = [
        f"{unique_sorted[k]}|{unique_sorted[k + 1]}" for k in range(K - 1)
    ]

    # --- Build X (with intercept for proper contrast coding, then drop it) ---
    rhs_formula = rhs.strip()
    mm_full = formulaic.model_matrix("~ " + rhs_formula, data)
    full_cols = list(mm_full.columns)
    # Drop the Intercept column — thresholds absorb it
    if "Intercept" in full_cols:
        keep = [c for c in full_cols if c != "Intercept"]
        mm = mm_full[keep]
    else:
        mm = mm_full
    term_names = list(mm.columns)
    X = np.asarray(mm, dtype=np.float64)
    n, p = X.shape

    # --- Build Z ---
    Z = build_joint_z_from_specs(specs, data)
    n_levels_list = [
        int(np.unique(group_array(spec, nw_data)).shape[0]) for spec in specs
    ]

    # --- Theta setup ---
    n_theta = sum(n_theta_for_spec(s.n_terms, s.correlated) for s in specs)
    theta_bounds = _build_theta_bounds(specs)

    if theta0 is None:
        theta0_arr = np.ones(n_theta)
    else:
        theta0_arr = np.asarray(theta0, dtype=np.float64)

    # --- Initial thresholds from marginal proportions ---
    cum_counts = np.zeros(K)
    for k in range(K):
        cum_counts[k] = np.sum(y_codes <= k)
    cum_props = cum_counts / n
    # Clip to avoid inf in link
    cum_props = np.clip(cum_props, 0.01, 0.99)
    # Initial thresholds: link(cumulative proportion)
    init_thresh = np.empty(K - 1)
    for k in range(K - 1):
        init_thresh[k] = float(
            np.log(cum_props[k] / (1.0 - cum_props[k]))
            if link == "logit"
            else cdf_fn(np.array([cum_props[k]]))[0]
        )

    # For probit/cloglog, use inverse CDF
    if link == "probit":
        from scipy.special import ndtri

        init_thresh = ndtri(cum_props[: K - 1])
    elif link == "cloglog":
        init_thresh = np.log(-np.log(1.0 - cum_props[: K - 1]))

    alpha1_init, log_deltas_init = _thresholds_to_increments(init_thresh)

    # --- Pack initial params: [alpha1, log_deltas..., theta...] ---
    n_thresh_params = K - 1  # alpha1 + (K-2) log_deltas
    params0 = np.concatenate([[alpha1_init], log_deltas_init, theta0_arr])

    # Bounds: thresholds are unbounded, theta has lower bounds
    thresh_bounds: list[tuple[float | None, float | None]] = [
        (None, None)
    ] * n_thresh_params
    all_bounds = thresh_bounds + list(theta_bounds)

    # --- Phase 1: Optimize (thresholds, theta), profile out beta via PIRLS ---
    warm: dict[str, np.ndarray | None] = {"u": None, "beta": None}

    def obj_phase1(params: np.ndarray) -> float:
        return _clmm_objective(
            params,
            n_thresh_params,
            p,
            y_codes,
            X,
            Z,
            specs,
            n_levels_list,
            cdf_fn,
            pdf_fn,
            pdf_deriv_fn,
            warm,
        )

    if optimizer == "bobyqa":
        import pybobyqa

        lower = np.array([lo if lo is not None else -1e10 for lo, _ in all_bounds])
        upper = np.array([hi if hi is not None else 1e10 for _, hi in all_bounds])
        soln = pybobyqa.solve(obj_phase1, params0, bounds=(lower, upper))
        params_hat = soln.x
        opt_converged = soln.msg == "Success: rho has reached rhoend"
    else:
        res = opt.minimize(
            obj_phase1,
            params0,
            method="L-BFGS-B",
            bounds=all_bounds,
        )
        params_hat = res.x
        opt_converged = bool(res.success)

    # Extract results from phase 1
    alpha1_hat = params_hat[0]
    log_deltas_hat = params_hat[1:n_thresh_params]
    theta_hat = params_hat[n_thresh_params:]
    thresholds_hat = _increments_to_thresholds(alpha1_hat, log_deltas_hat)

    # --- Phase 2: Joint (thresholds, beta, theta) via Nelder-Mead ---
    beta_phase1 = warm.get("beta")
    if beta_phase1 is None:
        beta_phase1 = np.zeros(p)

    params_phase2 = np.concatenate(
        [[alpha1_hat], log_deltas_hat, beta_phase1, theta_hat]
    )
    warm_phase2: dict[str, np.ndarray | None] = {"u": warm.get("u")}

    def obj_phase2(params: np.ndarray) -> float:
        return _clmm_joint_objective(
            params,
            n_thresh_params,
            p,
            y_codes,
            X,
            Z,
            specs,
            n_levels_list,
            cdf_fn,
            pdf_fn,
            pdf_deriv_fn,
            warm_phase2,
        )

    res2 = opt.minimize(
        obj_phase2,
        params_phase2,
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000, "adaptive": True},
    )

    phase1_ll = -obj_phase1(params_hat)
    phase2_ll = -res2.fun

    if phase2_ll > phase1_ll + 0.01:
        alpha1_hat = res2.x[0]
        log_deltas_hat = res2.x[1:n_thresh_params]
        beta_hat = res2.x[n_thresh_params : n_thresh_params + p]
        theta_hat = res2.x[n_thresh_params + p :]
        thresholds_hat = _increments_to_thresholds(alpha1_hat, log_deltas_hat)
        llf = phase2_ll
        warm["u"] = warm_phase2.get("u")
        warm["beta"] = beta_hat
        opt_converged = opt_converged and bool(res2.success)
    else:
        llf = phase1_ll
        beta_hat = warm.get("beta")
        if beta_hat is None:
            beta_hat = np.zeros(p)

    # --- Final PIRLS at optimum ---
    beta_final, u_hat, eta_final, final_ll, pirls_converged = _clmm_pirls(
        y_codes,
        X,
        Z,
        thresholds_hat,
        theta_hat,
        specs,
        n_levels_list,
        cdf_fn,
        pdf_fn,
        pdf_deriv_fn,
        u0=warm.get("u"),
        beta0=beta_hat,
    )
    llf = final_ll
    converged = opt_converged and pirls_converged

    # --- Standard errors via full observed information ---
    # R's ordinal::clmm computes SEs from the full information matrix
    # for (alpha, beta, theta), not conditioning on theta. Our SEs are
    # too narrow if we condition on theta because we ignore variance-
    # parameter uncertainty.
    #
    # Approach: numerical Hessian of the profiled Laplace LL w.r.t.
    # all parameters (alpha, beta, theta), then extract the (alpha, beta)
    # block from the inverse. PIRLS only solves for u.
    n_fe_total = (K - 1) + p
    n_all = n_fe_total + n_theta

    def _neg_ll_full(params: np.ndarray) -> float:
        """Neg Laplace LL as function of (alpha, beta, theta)."""
        thresh_h = params[: K - 1]
        beta_h = params[K - 1 : K - 1 + p]
        theta_h = params[K - 1 + p :]

        q_t = Z.shape[1]
        Lambda_h = make_lambda(theta_h, specs, n_levels_list)
        Z_star_h = Z @ Lambda_h
        u_h = u_hat.copy()

        for _it in range(_PIRLS_MAXITER):
            eta_h = X @ beta_h + Z @ u_h
            sc, nh = _ordinal_score_hessian_vec(
                y_codes,
                thresh_h,
                eta_h,
                cdf_fn,
                pdf_fn,
                pdf_deriv_fn,
            )
            z_w_h = eta_h + sc / nh
            residual = z_w_h - X @ beta_h
            sqW = np.sqrt(nh)
            WZs = sp.diags(sqW, format="csc") @ Z_star_h
            ZstWr = np.asarray(Z_star_h.T @ (nh * residual)).squeeze()
            A_h = ((WZs.T @ WZs) + sp.eye(q_t, format="csc")).tocsc()
            v_new = spla.spsolve(A_h, ZstWr)
            u_new = np.asarray(Lambda_h @ v_new).squeeze()
            if np.max(np.abs(u_new - u_h)) < _PIRLS_TOL:
                u_h = u_new
                break
            u_h = u_new

        eta_h = X @ beta_h + Z @ u_h
        cond_ll = _ordinal_loglik(y_codes, thresh_h, eta_h, cdf_fn)
        penalty = float(v_new @ v_new)
        _, nh_f = _ordinal_score_hessian_vec(
            y_codes,
            thresh_h,
            eta_h,
            cdf_fn,
            pdf_fn,
            pdf_deriv_fn,
        )
        WZs_f = sp.diags(np.sqrt(nh_f), format="csc") @ Z_star_h
        A_f = ((WZs_f.T @ WZs_f) + sp.eye(q_t, format="csc")).tocsc()
        log_det_A = sparse_chol_logdet(A_f)
        ll = cond_ll - 0.5 * penalty - 0.5 * log_det_A
        if not np.isfinite(ll):
            return 1e20
        return -ll

    # Numerical Hessian via central differences over all parameters
    all_params_hat = np.concatenate([thresholds_hat, beta_final, theta_hat])
    step = 1e-4
    H = np.zeros((n_all, n_all))
    for i in range(n_all):
        for j in range(i, n_all):
            ei = np.zeros(n_all)
            ej = np.zeros(n_all)
            ei[i] = step
            ej[j] = step
            fpp = _neg_ll_full(all_params_hat + ei + ej)
            fpm = _neg_ll_full(all_params_hat + ei - ej)
            fmp = _neg_ll_full(all_params_hat - ei + ej)
            fmm = _neg_ll_full(all_params_hat - ei - ej)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * step**2)
            H[j, i] = H[i, j]

    # Invert full Hessian, extract (alpha, beta) block
    try:
        cov_full = la.inv(H)
    except la.LinAlgError:
        cov_full = np.linalg.pinv(H)

    # SEs from the (alpha, beta) subblock of the full covariance
    cov_fe = cov_full[:n_fe_total, :n_fe_total]
    se_all = np.sqrt(np.maximum(np.diag(cov_fe), 0.0))
    threshold_se_arr = se_all[: K - 1]
    beta_se_arr = se_all[K - 1 :]

    # --- Package results ---
    thresholds_dict = dict(zip(threshold_labels, thresholds_hat.tolist(), strict=True))
    threshold_bse_dict = dict(
        zip(threshold_labels, threshold_se_arr.tolist(), strict=True)
    )

    fe_params = pd.Series(beta_final, index=term_names)
    fe_bse = pd.Series(beta_se_arr, index=term_names)

    # Random effects
    random_effects: dict[str, Any] = {}
    variance_components: dict[str, float] = {}
    ngroups: dict[str, int] = {}

    theta_idx = 0
    blup_offset = 0
    for spec, q_j in zip(specs, n_levels_list, strict=True):
        n_theta_j = n_theta_for_spec(spec.n_terms, spec.correlated)
        n_blups_j = spec.n_terms * q_j
        blup_block = u_hat[blup_offset : blup_offset + n_blups_j]
        uniques = sorted(np.unique(group_array(spec, nw_data)).tolist())

        if spec.n_terms == 1:
            random_effects[spec.group] = pd.Series(
                blup_block, index=uniques, name=spec.group
            )
            theta_j0 = theta_hat[theta_idx]
            variance_components[spec.group] = float(theta_j0**2)
        else:
            term_names_j = (["(Intercept)"] if spec.intercept else []) + list(
                spec.predictors
            )
            theta_j = theta_hat[theta_idx : theta_idx + n_theta_j]
            re_mat = blup_block.reshape(spec.n_terms, q_j).T
            random_effects[spec.group] = pd.DataFrame(
                re_mat, index=uniques, columns=term_names_j
            )
            p_j = spec.n_terms
            if spec.correlated:
                L_j = np.zeros((p_j, p_j))
                idx = 0
                for row in range(p_j):
                    for col in range(row + 1):
                        L_j[row, col] = theta_j[idx]
                        idx += 1
                cov_mat = L_j @ L_j.T
            else:
                cov_mat = np.diag(theta_j**2)
            variance_components[spec.group] = float(cov_mat[0, 0])

        ngroups[spec.group] = q_j
        theta_idx += n_theta_j
        blup_offset += n_blups_j

    # Information criteria
    nparams = (K - 1) + p + n_theta  # thresholds + betas + theta
    aic = -2.0 * llf + 2.0 * nparams
    bic = -2.0 * llf + np.log(n) * nparams

    return CLMMResult(
        thresholds=thresholds_dict,
        threshold_bse=threshold_bse_dict,
        fe_params=fe_params,
        fe_bse=fe_bse,
        random_effects=random_effects,
        variance_components=variance_components,
        theta=theta_hat,
        converged=converged,
        nobs=n,
        llf=float(llf),
        aic=float(aic),
        bic=float(bic),
        ngroups=ngroups,
        link=link,
        _formula=formula,
        _group_cols=[s.group for s in specs],
    )
