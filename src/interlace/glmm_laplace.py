"""GLMM estimation via Laplace approximation (PIRLS).

Implements the penalized iteratively reweighted least squares (PIRLS)
algorithm from Bates et al. (2015) for generalized linear mixed models.

The Laplace approximation to the marginal log-likelihood is:

    log p(y|theta) ≈ log p(y|u_hat,beta_hat) - 0.5*u_hat'*Lambda'*Lambda*u_hat
                     - 0.5*log|L_theta|^2

where u_hat, beta_hat are found by the inner PIRLS loop and L_theta is
the Cholesky factor of the penalized system.

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
import pandas as pd
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from interlace.formula import (
    groups_to_random_effects,
    parse_formula,
    parse_random_effects,
)
from interlace.glmm_family import (
    GaussianFamily,
    GLMMFamily,
    NegativeBinomial2Family,
    resolve_family,
)
from interlace.profiled_reml import (
    _build_theta_bounds,
    make_lambda,
    n_theta_for_spec,
    sparse_chol_logdet,
)
from interlace.sparse_z import build_joint_z_from_specs, group_array

if TYPE_CHECKING:
    from interlace.formula import RandomEffectSpec


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GLMMResult:
    """Result container for a fitted GLMM."""

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
    family: GLMMFamily
    ngroups: dict[str, int]
    scale: float  # dispersion (1.0 for binomial/Poisson)
    fittedvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    _formula: str = ""
    _group_cols: list[str] | None = None
    _eta: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # linear predictor (link scale)
    disp_params: pd.Series | None = None  # dispersion formula coefficients (log link)
    dispersion: np.ndarray | None = None  # per-observation dispersion values

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "response",
        include_re: bool = True,
    ) -> np.ndarray:
        """Predict from a fitted GLMM.

        Parameters
        ----------
        newdata :
            DataFrame to predict on. If ``None``, returns in-sample
            predictions.
        type :
            ``"response"`` (default) returns predictions on the response
            scale (mu). ``"link"`` returns predictions on the linear
            predictor scale (eta).
        include_re :
            If ``True`` (default), include BLUPs for known group levels.
            If ``False``, return population-level (fixed-effects only)
            predictions.

        Returns
        -------
        np.ndarray of shape (n_obs,)
        """
        if type not in ("response", "link"):
            msg = f"type must be 'response' or 'link', got {type!r}"
            raise ValueError(msg)

        if newdata is None:
            eta = self._eta
            if not include_re:
                # Recompute FE-only eta from stored formula
                eta = self._fe_only_eta()
        else:
            eta = self._predict_newdata(newdata, include_re)

        if type == "link":
            return np.asarray(eta)
        return np.asarray(self.family.linkinv(eta))

    def _fe_only_eta(self) -> np.ndarray:
        """Return fixed-effects-only linear predictor for in-sample data."""
        # fittedvalues = linkinv(eta), eta = X@beta + Z@u
        # FE-only: X@beta = eta - Z@u
        # Since we don't store X separately, reconstruct from
        # eta and RE contributions. But simpler: we can use
        # the formula to rebuild X from stored data. However we
        # don't store the data. Instead, compute from eta and BLUPs.
        # For in-sample, the simplest approach: not supported without
        # storing X. Return eta minus RE contribution.
        #
        # Actually, we can approximate: fittedvalues on response scale
        # is linkinv(eta), and we have eta stored. We just need to
        # subtract the RE contribution. But we don't have Z stored.
        # For now, raise if data isn't available.
        msg = (
            "In-sample include_re=False requires newdata. "
            "Pass the original data explicitly."
        )
        raise ValueError(msg)

    def _predict_newdata(self, newdata: Any, include_re: bool) -> np.ndarray:
        """Compute linear predictor for new data."""
        import formulaic
        import narwhals as nw

        nw_new = nw.from_native(newdata, eager_only=True)

        # Build X from formula
        fe_formula = self._formula.split("~", 1)[1].strip()
        X_mm = formulaic.model_matrix(fe_formula, nw_new)
        mm_cols = list(X_mm.columns)
        mm_arr = np.asarray(X_mm)

        # Reorder/pad columns to match fe_params order
        fe_cols = list(self.fe_params.index)
        if mm_cols != fe_cols:
            n_obs = mm_arr.shape[0]
            col_lookup = {c: mm_arr[:, i] for i, c in enumerate(mm_cols)}
            mm_arr = np.column_stack(
                [col_lookup.get(c, np.zeros(n_obs)) for c in fe_cols]
            )

        eta = mm_arr @ np.asarray(self.fe_params)

        if not include_re or self._group_cols is None:
            return np.asarray(eta)

        # Add BLUP contributions
        for col in self._group_cols:
            if col not in nw_new.columns:
                continue
            blup_re = self.random_effects.get(col)
            if blup_re is None:
                continue
            col_vals = nw_new[col].to_numpy()

            if isinstance(blup_re, pd.DataFrame):
                predictors = list(blup_re.columns[1:])
                contrib = np.zeros(len(col_vals))
                for i, level in enumerate(col_vals):
                    if level not in blup_re.index:
                        continue
                    blup_vec = blup_re.loc[level].to_numpy(dtype=float)
                    z_row = np.array(
                        [1.0] + [float(nw_new[p].to_numpy()[i]) for p in predictors]
                    )
                    contrib[i] = blup_vec @ z_row
                eta = eta + contrib
            elif isinstance(blup_re, pd.Series):
                lookup = blup_re.to_dict()
                contrib = np.array([lookup.get(v, 0.0) for v in col_vals], dtype=float)
                eta = eta + contrib

        return np.asarray(eta)


# ---------------------------------------------------------------------------
# PIRLS inner loop
# ---------------------------------------------------------------------------

_PIRLS_MAXITER = 100
_PIRLS_TOL = 1e-8
_MU_EPS = 1e-10  # clamp mu away from 0 boundary


def _clamp_mu(mu: np.ndarray, family: GLMMFamily) -> np.ndarray:
    """Clamp mu to valid range for the family."""
    if family.name == "binomial":
        return np.asarray(np.clip(mu, _MU_EPS, 1.0 - _MU_EPS))
    if family.name in ("poisson", "negativebinomial"):
        return np.asarray(np.maximum(mu, _MU_EPS))
    return mu


def _conditional_loglik(
    y: np.ndarray,
    mu: np.ndarray,
    weights: np.ndarray,
    family: GLMMFamily,
    phi: np.ndarray | None = None,
) -> float:
    """Compute the conditional log-likelihood sum_i log p(y_i | mu_i).

    For binomial (proportion y, trial count wt):
        ll_i = log(C(n_i, k_i)) + k_i*log(mu_i) + (n_i - k_i)*log(1 - mu_i)
        where k_i = y_i * n_i, n_i = wt_i.

    For Poisson (count y, wt=1):
        ll_i = y_i*log(mu_i) - mu_i - log(y_i!)

    For Gaussian with dispersion phi_i:
        ll_i = -0.5 * [log(phi_i) + (y_i - mu_i)^2 / phi_i] + const

    Parameters
    ----------
    phi : Per-observation dispersion vector.  ``None`` means dispersion = 1.
    """
    from scipy.special import gammaln

    if family.name == "binomial":
        n_trials = weights
        k = y * n_trials  # successes
        # log(C(n, k)) + k*log(mu) + (n-k)*log(1-mu)
        log_binom = gammaln(n_trials + 1) - gammaln(k + 1) - gammaln(n_trials - k + 1)
        mu_safe = _clamp_mu(mu, family)
        ll = log_binom + k * np.log(mu_safe) + (n_trials - k) * np.log(1.0 - mu_safe)
        return float(np.sum(ll))
    elif family.name == "poisson":
        mu_safe = np.maximum(mu, _MU_EPS)
        ll = y * np.log(mu_safe) - mu_safe - gammaln(y + 1)
        return float(np.sum(weights * ll))
    elif family.name == "negativebinomial":
        assert isinstance(family, NegativeBinomial2Family)
        theta = family.theta
        mu_safe = np.maximum(mu, _MU_EPS)
        # NB2 log-likelihood:
        # ll_i = lgamma(y+theta) - lgamma(theta) - lgamma(y+1)
        #        + theta*log(theta) - theta*log(mu+theta)
        #        + y*log(mu) - y*log(mu+theta)
        ll = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta)
            - theta * np.log(mu_safe + theta)
            + y * np.log(mu_safe)
            - y * np.log(mu_safe + theta)
        )
        return float(np.sum(weights * ll))
    elif family.name == "gaussian":
        n = len(y)
        if phi is not None:
            # Heteroscedastic: -0.5 * sum[log(phi_i) + (y-mu)^2/phi_i] + const
            ll = -0.5 * np.sum(
                np.log(phi) + weights * (y - mu) ** 2 / phi
            ) - 0.5 * n * np.log(2.0 * np.pi)
        else:
            # Homoscedastic (phi = 1)
            ll = -0.5 * np.sum(weights * (y - mu) ** 2) - 0.5 * n * np.log(2.0 * np.pi)
        return float(ll)
    else:
        # Fallback: use -0.5 * deviance (no normalizing constant)
        dev = float(np.sum(family.dev_resids(y, mu, weights)))
        return -0.5 * dev


def _glm_start(
    y: np.ndarray,
    X: np.ndarray,
    family: GLMMFamily,
    weights: np.ndarray,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Compute starting beta from a fixed-effects-only GLM (IRLS).

    Runs a few IRLS iterations without random effects to get a reasonable
    starting point for PIRLS.
    """
    n, p = X.shape
    _off = offset if offset is not None else np.zeros(n)

    # Initialize mu from y, with safety clamps
    if family.name == "binomial":
        mu = np.clip(y, 0.01, 0.99)
    elif family.name in ("poisson", "negativebinomial"):
        mu = np.maximum(y, 0.1)
    else:
        mu = y.copy()

    beta = np.zeros(p)
    for _ in range(25):
        eta = family.link(mu)
        mu_eta_val = family.mu_eta(eta)
        var_mu = family.variance(mu)
        w = weights * mu_eta_val**2 / var_mu
        # Working residual on the link scale, excluding offset
        z_w = (eta - _off) + (y - mu) / mu_eta_val

        WX = np.sqrt(w)[:, None] * X
        Wz = np.sqrt(w) * z_w
        try:
            beta_new = la.solve(WX.T @ WX, WX.T @ Wz, assume_a="pos")
        except la.LinAlgError:
            break
        eta = X @ beta_new + _off
        mu = family.linkinv(eta)
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    return beta


def _pirls(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    theta: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    u0: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    phi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Run PIRLS to find conditional modes (u_hat, beta_hat).

    Parameters
    ----------
    y : Response vector (n,). For binomial, this is proportion (0-1).
    X : Fixed-effects design matrix (n, p).
    Z : Random-effects design matrix (n, q).
    family : GLMMFamily instance.
    theta : Variance parameters.
    specs : Random effect specifications.
    n_levels : Number of levels per grouping factor.
    weights : Prior weights (n,). For binomial, this is the trial count.
    u0 : Initial random effects (q,). Defaults to zeros.
    beta0 : Initial fixed effects (p,). Defaults to zeros.
    offset : Offset vector (n,). Added to the linear predictor.
    phi : Per-observation dispersion vector (n,).  ``None`` means 1.

    Returns
    -------
    beta_hat : Fixed effects (p,).
    u_hat : Conditional modes of random effects (q,).
    mu_hat : Fitted means on response scale (n,).
    laplace_loglik : Laplace-approximated log-likelihood.
    converged : Whether PIRLS converged.
    """
    n, p = X.shape
    q = Z.shape[1]

    Lambda = make_lambda(theta, specs, n_levels)

    u = np.zeros(q) if u0 is None else u0.copy()
    # Initialize beta from a GLM fit (no random effects) when not warm-starting.
    beta = (
        _glm_start(y, X, family, weights, offset=offset)
        if beta0 is None
        else beta0.copy()
    )

    _off = offset if offset is not None else np.zeros(n)

    converged = False

    for _iteration in range(_PIRLS_MAXITER):
        # Current linear predictor and mean
        eta = X @ beta + Z @ u + _off
        mu = family.linkinv(eta)

        # Clamp mu to avoid numerical issues
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)

        # Working weights and working residual
        mu_eta_val = family.mu_eta(eta)  # d(mu)/d(eta)
        var_mu = family.variance(mu)

        # W = diag(weights * mu_eta^2 / (phi * var_mu)) — the IRLS weight matrix
        # When phi is provided, it scales the variance function.
        denom = var_mu if phi is None else phi * var_mu
        w = weights * mu_eta_val**2 / denom  # (n,)
        # Working residual (offset excluded so we solve for X@beta + Z@u only)
        z_w = (eta - _off) + (y - mu) / mu_eta_val  # (n,)

        # Penalized weighted least squares via Lambda parameterisation.
        # Let v = Lambda^{-1} u so the penalty term becomes v'v.

        sqrtW = np.sqrt(w)  # (n,)
        # Scale everything by sqrt(W)
        WX = sqrtW[:, None] * X  # (n, p)
        Wz = sqrtW * z_w  # (n,)

        # Lambda parameterisation: v = Lambda^{-1} u, penalty = v'v
        Z_star = Z @ Lambda  # (n, q)
        WZs = sp.diags(sqrtW, format="csc") @ Z_star  # (n, q)

        XtWX = WX.T @ WX  # (p, p)
        ZstWX = np.asarray(
            (WZs.T @ WX).toarray() if sp.issparse(WZs.T @ WX) else WZs.T @ WX
        )  # (q, p)
        ZstWZs = (WZs.T @ WZs).tocsc()  # (q, q)
        XtWz = WX.T @ Wz  # (p,)
        ZstWz = np.asarray(WZs.T @ Wz).squeeze()  # (q,)

        # A = Z_star'WZ_star + I
        A = (ZstWZs + sp.eye(q, format="csc")).tocsc()

        # Schur complement solve for beta, then back-substitute for v
        A_inv_ZstWX = np.column_stack(
            [spla.spsolve(A, ZstWX[:, j]) for j in range(p)]
        )  # (q, p)
        A_inv_ZstWz = spla.spsolve(A, ZstWz)  # (q,)

        schur = XtWX - ZstWX.T @ A_inv_ZstWX  # (p, p)
        rhs_beta = XtWz - ZstWX.T @ A_inv_ZstWz  # (p,)

        try:
            beta_new = la.solve(schur, rhs_beta, assume_a="pos")
        except la.LinAlgError:
            beta_new = la.lstsq(schur, rhs_beta)[0]

        v_new = A_inv_ZstWz - A_inv_ZstWX @ beta_new  # (q,)
        u_new = np.asarray(Lambda @ v_new).squeeze()  # (q,)

        # Step-halving: limit the maximum change per iteration to prevent
        # overshooting in Poisson/binomial models with extreme predictions.
        max_step = 5.0
        delta_beta_raw = beta_new - beta
        delta_u_raw = u_new - u
        max_delta = max(np.max(np.abs(delta_beta_raw)), np.max(np.abs(delta_u_raw)))
        if max_delta > max_step:
            scale = max_step / max_delta
            beta_new = beta + scale * delta_beta_raw
            u_new = u + scale * delta_u_raw

        # Check convergence
        delta_beta = np.max(np.abs(beta_new - beta))
        delta_u = np.max(np.abs(u_new - u))
        max_change = max(delta_beta, delta_u)

        beta = beta_new
        u = u_new

        if max_change < _PIRLS_TOL:
            converged = True
            break

    # Final values
    eta = X @ beta + Z @ u + _off
    mu = family.linkinv(eta)
    if not isinstance(family, GaussianFamily):
        mu = _clamp_mu(mu, family)

    # Use v from the last PIRLS iteration (Lambda^{-1} u)
    v_final = v_new if converged or _iteration > 0 else np.zeros(q)  # noqa: F821
    penalty = float(v_final @ v_final)

    # Recompute A at final values for log|A|
    mu_eta_val = family.mu_eta(eta)
    var_mu = family.variance(mu)
    denom_final = var_mu if phi is None else phi * var_mu
    w = weights * mu_eta_val**2 / denom_final
    WZs_final = sp.diags(np.sqrt(w), format="csc") @ Z_star
    A_final = ((WZs_final.T @ WZs_final) + sp.eye(q, format="csc")).tocsc()
    log_det_A = sparse_chol_logdet(A_final)

    # Laplace log-likelihood:
    # ll = log p(y|u_hat, beta_hat) - 0.5*v'v - 0.5*log|A|
    # where log p(y|...) is the conditional log-likelihood (not deviance).
    cond_ll = _conditional_loglik(y, mu, weights, family, phi=phi)

    laplace_ll = cond_ll - 0.5 * penalty - 0.5 * log_det_A

    return beta, u, mu, laplace_ll, converged


# ---------------------------------------------------------------------------
# Outer optimisation over theta
# ---------------------------------------------------------------------------


def _laplace_objective(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    warm: dict[str, np.ndarray | None],
    offset: np.ndarray | None = None,
    phi: np.ndarray | None = None,
) -> float:
    """Negative Laplace log-likelihood (to minimize over theta)."""
    beta, u, _mu, ll, _conv = _pirls(
        y,
        X,
        Z,
        family,
        theta,
        specs,
        n_levels,
        weights,
        u0=warm.get("u"),
        beta0=warm.get("beta"),
        offset=offset,
        phi=phi,
    )
    # Warm-start next call
    warm["u"] = u
    warm["beta"] = beta

    if not np.isfinite(ll):
        return 1e20
    return -ll


# ---------------------------------------------------------------------------
# AGQ (Adaptive Gauss-Hermite Quadrature) objective
# ---------------------------------------------------------------------------


def _agq_loglik(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    nAGQ: int,
    group_indices: list[np.ndarray],
    warm: dict[str, np.ndarray | None],
    offset: np.ndarray | None = None,
) -> float:
    """Compute the AGQ-approximated marginal log-likelihood.

    For each group i the marginal contribution is estimated by adaptive
    Gauss-Hermite quadrature over the scalar random intercept u_i, adapting
    the quadrature nodes to the conditional mode and curvature found by PIRLS.

    Parameters
    ----------
    theta : Variance parameters.
    y, X, Z : Model matrices.
    family : GLMMFamily instance.
    specs : Random effect specifications (must be single scalar intercept).
    n_levels : Number of levels per grouping factor.
    weights : Prior weights.
    nAGQ : Number of GH quadrature points.
    group_indices : List of arrays, one per group level, each containing
        the row indices of observations belonging to that level.
    warm : Warm-start dict for PIRLS.
    offset : Offset vector (n,). Added to the linear predictor.

    Returns
    -------
    Negative log-likelihood (for minimisation).
    """
    from numpy.polynomial.hermite import hermgauss

    # Step 1: PIRLS to find conditional modes and working quantities
    beta, u, _mu, _laplace_ll, _conv = _pirls(
        y,
        X,
        Z,
        family,
        theta,
        specs,
        n_levels,
        weights,
        u0=warm.get("u"),
        beta0=warm.get("beta"),
        offset=offset,
    )
    warm["u"] = u
    warm["beta"] = beta

    _off = offset if offset is not None else np.zeros(len(y))

    sigma_u = float(theta[0])  # theta parameterises SD: sigma_u = theta * sigma
    # For GLMM dispersion is 1 (binomial/Poisson), so var_u = theta^2
    var_u = sigma_u**2

    if var_u < 1e-20:
        # Degenerate: no random effects, fall back to conditional ll
        eta = X @ beta + _off
        mu = family.linkinv(eta)
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)
        cond_ll = _conditional_loglik(y, mu, weights, family)
        return -cond_ll if np.isfinite(cond_ll) else 1e20

    # Step 2: Compute conditional precision per group from PIRLS working weights
    eta_hat = X @ beta + Z @ u + _off
    mu_hat = family.linkinv(eta_hat)
    if not isinstance(family, GaussianFamily):
        mu_hat = _clamp_mu(mu_hat, family)

    mu_eta_hat = family.mu_eta(eta_hat)
    var_mu_hat = family.variance(mu_hat)
    w_hat = weights * mu_eta_hat**2 / var_mu_hat  # IRLS weights at mode

    # GH nodes and weights
    gh_z, gh_w = hermgauss(nAGQ)  # ∫ exp(-t²) f(t) dt ≈ Σ w_k f(z_k)

    q = n_levels[0]  # number of groups
    total_ll = 0.0

    for i in range(q):
        idx = group_indices[i]
        u_hat_i = float(u[i])

        # Conditional precision: h_i = Σ_j w_{ij} + 1/var_u
        h_i = float(np.sum(w_hat[idx])) + 1.0 / var_u
        sigma_c_i = 1.0 / np.sqrt(h_i)  # conditional SD

        # Pre-extract group data
        y_i = y[idx]
        X_i = X[idx]
        w_i = weights[idx]
        off_i = _off[idx]

        # For each GH node, compute log-integrand
        # u_k = u_hat_i + sqrt(2) * sigma_c_i * z_k
        log_integrands = np.empty(nAGQ)
        for k in range(nAGQ):
            u_ik = u_hat_i + np.sqrt(2.0) * sigma_c_i * gh_z[k]

            # Linear predictor for group i at this u
            eta_ik = X_i @ beta + u_ik + off_i
            mu_ik = family.linkinv(eta_ik)
            if not isinstance(family, GaussianFamily):
                mu_ik = _clamp_mu(mu_ik, family)

            # Conditional log-likelihood for group i
            cll_ik = _conditional_loglik(y_i, mu_ik, w_i, family)

            # Log prior: log N(u_ik; 0, var_u)
            log_prior_ik = -0.5 * u_ik**2 / var_u

            # g(u_ik) = cond_ll + log_prior (unnormalised)
            g_ik = cll_ik + log_prior_ik

            # Integrand for GH: exp(g(u_ik) + z_k^2) * w_k
            # We work on log scale: log(w_k) + g_ik + z_k^2
            log_integrands[k] = np.log(gh_w[k]) + g_ik + gh_z[k] ** 2

        # log L_i = log(sqrt(2) * sigma_c_i) + logsumexp(log_integrands)
        max_li = np.max(log_integrands)
        log_Li = (
            0.5 * np.log(2.0)
            + np.log(sigma_c_i)
            + max_li
            + np.log(np.sum(np.exp(log_integrands - max_li)))
        )
        total_ll += log_Li

    # Subtract the log-prior normalising constant: -0.5*q*log(2*pi*var_u)
    # (the N(0, var_u) density has normalisation 1/sqrt(2*pi*var_u) per group)
    total_ll -= 0.5 * q * np.log(2.0 * np.pi * var_u)

    if not np.isfinite(total_ll):
        return 1e20
    return float(-total_ll)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_glmm(
    formula: str,
    data: Any,
    family: str | GLMMFamily,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    weights: np.ndarray | None = None,
    optimizer: str = "lbfgsb",
    theta0: np.ndarray | None = None,
    nAGQ: int = 1,
    offset: np.ndarray | None = None,
    dispformula: str | None = None,
) -> GLMMResult:
    """Fit a generalized linear mixed model.

    Parameters
    ----------
    formula :
        Fixed-effects formula, e.g. ``"y ~ x1 + x2"``.
    data :
        DataFrame (pandas, polars, or any narwhals-compatible frame).
    family :
        Family name (``"binomial"``, ``"poisson"``, ``"gaussian"``) or
        a :class:`GLMMFamily` instance.
    groups :
        Column name(s) for random intercepts.
    random :
        lme4-style random effect specs (takes precedence over ``groups``).
    weights :
        Prior weights. For binomial proportion response, pass trial counts.
        Defaults to ones.
    optimizer :
        ``"lbfgsb"`` (default) or ``"bobyqa"``.
    theta0 :
        Initial theta. Defaults to ones.
    nAGQ :
        Number of adaptive Gauss-Hermite quadrature points.  ``1`` (default)
        uses the Laplace approximation.  Values ``> 1`` use AGQ for a more
        accurate marginal-likelihood integral; this requires a single scalar
        random intercept (one grouping factor, intercept only).
    offset :
        Offset vector, shape ``(n,)``.  A known term added to the linear
        predictor that is not estimated.  Common use: ``np.log(exposure)``
        in Poisson rate models.  Defaults to zero.
    dispformula :
        Formula for the dispersion sub-model with a log link, e.g.
        ``"~1"`` for scalar dispersion or ``"~ z"`` for covariate-dependent
        dispersion.  The dispersion for observation *i* is
        ``phi_i = exp(X_d[i] @ delta)``.  ``None`` (default) fixes
        dispersion at 1.0.  Cannot be combined with ``nAGQ > 1``.

    Returns
    -------
    GLMMResult
    """
    import formulaic
    import narwhals as nw

    fam = resolve_family(family)
    nw_data = nw.from_native(data, eager_only=True)

    # --- Build random effect specs ---
    if random is not None:
        specs = parse_random_effects(random)
    elif groups is not None:
        specs = groups_to_random_effects(groups)
    else:
        raise ValueError("Either 'groups' or 'random' must be provided.")

    group_cols = [s.group for s in specs]

    # --- Validate nAGQ ---
    if nAGQ < 1:
        raise ValueError("nAGQ must be >= 1.")
    if nAGQ > 1:
        if len(specs) > 1:
            msg = "nAGQ > 1 is only supported with a single grouping factor."
            raise ValueError(msg)
        if specs[0].n_terms > 1:
            msg = (
                "nAGQ > 1 is only supported for scalar random intercepts "
                "(no random slopes)."
            )
            raise ValueError(msg)

    # --- Validate dispformula + nAGQ ---
    if dispformula is not None and nAGQ > 1:
        msg = "dispformula cannot be combined with nAGQ > 1."
        raise ValueError(msg)

    # --- Parse formula, build X ---
    parsed = parse_formula(formula, data, groups=group_cols[0])
    y = parsed.y
    X = parsed.X
    term_names = parsed.term_names
    n, p = X.shape

    # --- Build dispersion design matrix ---
    if dispformula is not None:
        disp_formula_rhs = dispformula.lstrip("~").strip()
        if not disp_formula_rhs:
            disp_formula_rhs = "1"
        X_d_mm = formulaic.model_matrix("~ " + disp_formula_rhs, nw_data)
        disp_term_names = list(X_d_mm.columns)
        X_d = np.asarray(X_d_mm, dtype=np.float64)
        n_disp = X_d.shape[1]
    else:
        X_d = None
        disp_term_names = None
        n_disp = 0

    # --- Build Z ---
    Z = build_joint_z_from_specs(specs, data)
    n_levels_list = [
        int(np.unique(group_array(spec, nw_data)).shape[0]) for spec in specs
    ]

    # --- Weights ---
    if weights is None:
        weights_arr = np.ones(n)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)

    # --- Offset ---
    if offset is not None:
        offset_arr = np.asarray(offset, dtype=np.float64)
        if offset_arr.shape != (n,):
            msg = f"offset length ({offset_arr.size}) must match data length ({n})."
            raise ValueError(msg)
    else:
        offset_arr = np.zeros(n)

    # --- Theta setup ---
    n_theta = sum(n_theta_for_spec(s.n_terms, s.correlated) for s in specs)
    theta_bounds = _build_theta_bounds(specs)

    if theta0 is None:
        theta0 = np.ones(n_theta)

    # --- Optimize ---
    warm: dict[str, np.ndarray | None] = {"u": None, "beta": None}

    # Precompute group indices for AGQ
    if nAGQ > 1:
        group_codes = group_array(specs[0], nw_data)
        group_uniques = sorted(np.unique(group_codes).tolist())
        group_indices = [np.where(group_codes == lvl)[0] for lvl in group_uniques]

    if X_d is not None:
        # Joint optimization over [theta | delta].
        # delta are the dispersion regression coefficients (log link, unbounded).
        delta0 = np.zeros(n_disp)
        params0 = np.concatenate([theta0, delta0])
        joint_bounds = list(theta_bounds) + [(None, None)] * n_disp

        def joint_obj(params: np.ndarray) -> float:
            theta = params[:n_theta]
            delta = params[n_theta:]
            phi = np.exp(X_d @ delta)
            return _laplace_objective(
                theta,
                y,
                X,
                Z,
                fam,
                specs,
                n_levels_list,
                weights_arr,
                warm,
                offset=offset_arr,
                phi=phi,
            )

        lower_bounds = np.array(
            [lo if lo is not None else -np.inf for lo, _ in joint_bounds]
        )

        if optimizer == "bobyqa":
            import pybobyqa

            upper = np.array(
                [hi if hi is not None else np.inf for _, hi in joint_bounds]
            )
            soln = pybobyqa.solve(joint_obj, params0, bounds=(lower_bounds, upper))
            params_hat = soln.x
            opt_converged = soln.msg == "Success: rho has reached rhoend"
        else:
            res = opt.minimize(
                joint_obj,
                params0,
                method="L-BFGS-B",
                bounds=joint_bounds,
            )
            params_hat = res.x
            opt_converged = bool(res.success)

        theta_hat = params_hat[:n_theta]
        delta_hat = params_hat[n_theta:]
        phi_hat = np.exp(X_d @ delta_hat)
    else:
        # No dispformula — original optimization path
        delta_hat = None
        phi_hat = None

        def obj(theta: np.ndarray) -> float:
            if nAGQ > 1:
                return _agq_loglik(
                    theta,
                    y,
                    X,
                    Z,
                    fam,
                    specs,
                    n_levels_list,
                    weights_arr,
                    nAGQ,
                    group_indices,
                    warm,
                    offset=offset_arr,
                )
            return _laplace_objective(
                theta,
                y,
                X,
                Z,
                fam,
                specs,
                n_levels_list,
                weights_arr,
                warm,
                offset=offset_arr,
            )

        lower_bounds = np.array(
            [lo if lo is not None else -np.inf for lo, _ in theta_bounds]
        )

        if optimizer == "bobyqa":
            import pybobyqa

            upper = np.array(
                [hi if hi is not None else np.inf for _, hi in theta_bounds]
            )
            soln = pybobyqa.solve(obj, theta0, bounds=(lower_bounds, upper))
            theta_hat = soln.x
            opt_converged = soln.msg == "Success: rho has reached rhoend"
        else:
            res = opt.minimize(
                obj,
                theta0,
                method="L-BFGS-B",
                bounds=theta_bounds,
            )
            theta_hat = res.x
            opt_converged = bool(res.success)

    # --- Final PIRLS at optimum ---
    beta_hat, u_hat, mu_hat, laplace_llf, pirls_converged = _pirls(
        y,
        X,
        Z,
        fam,
        theta_hat,
        specs,
        n_levels_list,
        weights_arr,
        u0=warm.get("u"),
        beta0=warm.get("beta"),
        offset=offset_arr,
        phi=phi_hat,
    )
    converged = opt_converged and pirls_converged

    # For AGQ, recompute the final log-likelihood using AGQ at theta_hat
    if nAGQ > 1:
        warm_final: dict[str, np.ndarray | None] = {"u": u_hat, "beta": beta_hat}
        llf = -_agq_loglik(
            theta_hat,
            y,
            X,
            Z,
            fam,
            specs,
            n_levels_list,
            weights_arr,
            nAGQ,
            group_indices,
            warm_final,
            offset=offset_arr,
        )
    else:
        llf = laplace_llf

    # --- Fixed effects standard errors ---
    # From the Hessian of the penalized log-likelihood w.r.t. beta
    eta = X @ beta_hat + Z @ u_hat + offset_arr
    mu_eta_val = fam.mu_eta(eta)
    var_mu = fam.variance(mu_hat)
    denom_se = var_mu if phi_hat is None else phi_hat * var_mu
    w = weights_arr * mu_eta_val**2 / denom_se
    WX = np.sqrt(w)[:, None] * X
    XtWX = WX.T @ WX

    Lambda = make_lambda(theta_hat, specs, n_levels_list)
    Z_star = Z @ Lambda
    WZs = sp.diags(np.sqrt(w), format="csc") @ Z_star
    ZstWZs = (WZs.T @ WZs).tocsc()
    q = Z.shape[1]
    A = (ZstWZs + sp.eye(q, format="csc")).tocsc()
    ZstWX = np.asarray(
        (WZs.T @ WX).toarray() if sp.issparse(WZs.T @ WX) else WZs.T @ WX
    )
    A_inv_ZstWX = np.column_stack([spla.spsolve(A, ZstWX[:, j]) for j in range(p)])
    schur = XtWX - ZstWX.T @ A_inv_ZstWX  # Marginal precision of beta

    try:
        fe_cov = la.inv(schur)
    except la.LinAlgError:
        fe_cov = np.linalg.pinv(schur)

    fe_bse_arr = np.sqrt(np.maximum(np.diag(fe_cov), 0.0))

    # --- Package results ---
    fe_params = pd.Series(beta_hat, index=term_names)
    fe_bse = pd.Series(fe_bse_arr, index=term_names)

    # --- Random effects per spec ---
    random_effects: dict[str, Any] = {}
    variance_components: dict[str, float] = {}
    ngroups: dict[str, int] = {}
    sigma2 = 1.0  # dispersion fixed at 1 for binomial/Poisson

    theta_idx = 0
    blup_offset = 0
    for spec, q_j in zip(specs, n_levels_list, strict=True):
        n_theta_j = n_theta_for_spec(spec.n_terms, spec.correlated)
        n_blups_j = spec.n_terms * q_j
        blup_block = u_hat[blup_offset : blup_offset + n_blups_j]
        uniques = sorted(np.unique(group_array(spec, nw_data)).tolist())

        if spec.n_terms == 1:
            random_effects[spec.group] = pd.Series(
                blup_block,
                index=uniques,
                name=spec.group,
            )
            theta_j0 = theta_hat[theta_idx]
            variance_components[spec.group] = float(sigma2 * theta_j0**2)
        else:
            term_names_j = (["(Intercept)"] if spec.intercept else []) + list(
                spec.predictors
            )
            theta_j = theta_hat[theta_idx : theta_idx + n_theta_j]
            re_mat = blup_block.reshape(spec.n_terms, q_j).T
            random_effects[spec.group] = pd.DataFrame(
                re_mat,
                index=uniques,
                columns=term_names_j,
            )
            p_j = spec.n_terms
            if spec.correlated:
                L_j = np.zeros((p_j, p_j))
                idx = 0
                for row in range(p_j):
                    for col in range(row + 1):
                        L_j[row, col] = theta_j[idx]
                        idx += 1
                cov_mat = sigma2 * L_j @ L_j.T
            else:
                cov_mat = np.diag(sigma2 * theta_j**2)
            variance_components[spec.group] = float(cov_mat[0, 0])

        ngroups[spec.group] = q_j
        theta_idx += n_theta_j
        blup_offset += n_blups_j

    # --- Dispersion results ---
    if delta_hat is not None:
        disp_params = pd.Series(delta_hat, index=disp_term_names)
    else:
        disp_params = None

    # --- Information criteria ---
    nparams = p + n_theta + n_disp
    aic = -2.0 * llf + 2.0 * nparams
    bic = -2.0 * llf + np.log(n) * nparams

    # --- Fitted values (response scale) and linear predictor ---
    eta_hat = X @ beta_hat + Z @ u_hat + offset_arr
    fittedvalues = np.asarray(fam.linkinv(eta_hat))

    return GLMMResult(
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
        family=fam,
        ngroups=ngroups,
        scale=sigma2,
        fittedvalues=fittedvalues,
        _formula=formula,
        _group_cols=group_cols,
        _eta=np.asarray(eta_hat),
        disp_params=disp_params,
        dispersion=phi_hat,
    )
