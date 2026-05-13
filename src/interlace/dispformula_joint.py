"""Joint Laplace fit for heteroscedastic LMM with random effects on
``dispformula`` (interlace-c803).

Model
-----
    y_i | b, u_d  ~  N(X_i β + Z_i b, σ_i²),    log σ_i = W_i δ + V_i u_d
    b   = Λ_m(θ_m) v_m,   v_m ~ N(0, I)
    u_d = Λ_d(θ_d) v_d,   v_d ~ N(0, I)

Marginal Laplace log-likelihood
-------------------------------
    ℓ_marg(θ_m, θ_d, δ) ≈ ℓ_cond(β̂, v̂_m, v̂_d)
                          − ½‖v̂_m‖² − ½‖v̂_d‖²
                          − ½ log|H_joint/(2π)|

with H_joint = block(H_mm, H_dd) and the off-block H_md ≈ 0 in expectation
(the score Σ V_ig e_i/σ_i² has mean zero by symmetry of the Gaussian
residuals). We use the block-diagonal approximation log|H_joint| ≈
log|H_mm| + log|H_dd|; the cross-term contribution is O(1/√n) and vanishes
at the joint MLE for the variance components.

The inner mode is found by alternating two Newton-like steps:
    (i)  (β, v_m) given v_d  — exact one-shot weighted-LMM solve, since the
         Gaussian-heteroscedastic conditional is quadratic in (β, b);
    (ii) v_d given (β, v_m)  — penalised Newton step on the conditional
         log-likelihood for the dispersion latent vector.

This gives the same joint mode as block-coordinate ascent over (β, v_m, v_d).
The crucial differentiator vs the BCA path in :mod:`dispformula_bca` is the
**outer** objective: BCA's outer loop optimises an LMM REML criterion (for
θ_m) and a Gamma-GLMM Laplace (for θ_d) separately — neither corresponds
to the joint marginal log-likelihood. Here we instead optimise a single
unified ℓ_marg over (θ_m, θ_d, δ), restoring parity with glmmTMB.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from interlace.dispformula_bca import (
    _materialize_interaction_columns,
    _parse_dispformula,
    _parse_dispformula_raw,
    _to_pandas,
)
from interlace.formula import (
    groups_to_random_effects,
    parse_formula,
    parse_random_effects,
)
from interlace.profiled_reml import (
    LambdaBuilder,
    _build_theta_bounds,
)
from interlace.result import CrossedLMEResult, ModelInfo, _DataWrapper
from interlace.sparse_z import build_joint_z_from_specs, group_array

_INNER_MAXITER = 80
_INNER_TOL = 1e-7
_PHASE_TOL = 1e-7


def fit_dispformula_joint(
    formula: str,
    data: Any,
    dispformula: str,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    method: str = "REML",
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    df_method: str = "satterthwaite",
) -> Any:
    """Joint Laplace fit of Gaussian LMM with RE-on-dispformula."""
    # --- 1. Parse mean side ---
    if random is not None:
        specs_m = parse_random_effects(random)
    elif groups is not None:
        specs_m = groups_to_random_effects(groups)
    else:
        raise ValueError("Either 'groups' or 'random' must be provided.")

    parsed = parse_formula(formula, data, groups=specs_m[0].group)
    y = np.asarray(parsed.y, dtype=np.float64)
    X = np.asarray(parsed.X, dtype=np.float64)
    term_names = list(parsed.term_names)
    n, p = X.shape

    nw_data = nw.from_native(data, eager_only=True)
    Z_m_raw = build_joint_z_from_specs(specs_m, data)
    Z_m = Z_m_raw.tocsc() if sp.issparse(Z_m_raw) else sp.csc_matrix(Z_m_raw)
    qm = int(Z_m.shape[1])
    nlev_m = [int(np.unique(group_array(s, nw_data)).shape[0]) for s in specs_m]

    # --- 2. Parse dispformula ---
    disp_fe_rhs, disp_re_specs = _parse_dispformula(dispformula)
    _, _disp_re_raw = _parse_dispformula_raw(dispformula)
    if not disp_re_specs:
        raise ValueError(
            "fit_dispformula_joint requires random effects on the dispformula "
            "side; for FE-only dispformula use fit_dispformula_joint_laplace."
        )

    df_pd = _to_pandas(data)
    df_pd = _materialize_interaction_columns(df_pd, disp_re_specs)
    nw_df_pd = nw.from_native(df_pd, eager_only=True)

    import formulaic

    W_mm = formulaic.model_matrix("~ " + disp_fe_rhs, df_pd)
    disp_term_names = list(W_mm.columns)
    W = np.asarray(W_mm, dtype=np.float64)
    p_d = W.shape[1]

    Z_d_raw = build_joint_z_from_specs(disp_re_specs, df_pd)
    Z_d = Z_d_raw.tocsc() if sp.issparse(Z_d_raw) else sp.csc_matrix(Z_d_raw)
    qd = int(Z_d.shape[1])
    nlev_d = [int(np.unique(group_array(s, nw_df_pd)).shape[0]) for s in disp_re_specs]

    base_weights = (
        np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    )
    offset_arr = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)

    lb_m = LambdaBuilder(specs_m, nlev_m)
    lb_d = LambdaBuilder(disp_re_specs, nlev_d)
    n_th_m = lb_m._n_theta
    n_th_d = lb_d._n_theta

    # --- 3. Warm-start from BCA ---
    from interlace.dispformula_bca import fit_dispformula_bca

    bca = fit_dispformula_bca(
        formula=formula,
        data=df_pd,
        dispformula=dispformula,
        groups=groups,
        random=random,
        method=method,
        weights=weights,
        offset=offset,
        df_method=df_method,
    )

    theta_m0 = np.asarray(bca.theta, dtype=np.float64).copy()
    if theta_m0.size != n_th_m:
        theta_m0 = np.ones(n_th_m)
    delta0 = np.asarray(bca.disp_params.values, dtype=np.float64).copy()
    # rough theta_d init: sqrt of varcomp values, mapped to corresponding spec
    theta_d0 = np.ones(n_th_d)
    vc_d_warm = list(bca.disp_variance_components.values())
    if len(vc_d_warm) == n_th_d:
        theta_d0 = np.array(
            [np.sqrt(max(v, 1e-3)) for v in vc_d_warm], dtype=np.float64
        )

    beta0 = np.asarray(bca.fe_params.values, dtype=np.float64).copy()

    warm: dict[str, np.ndarray | None] = {
        "beta": beta0,
        "v_m": None,
        "v_d": None,
    }

    bounds = (
        _build_theta_bounds(specs_m)
        + _build_theta_bounds(disp_re_specs)
        + [(None, None)] * p_d
    )

    def outer_obj(params: np.ndarray) -> float:
        theta_m = params[:n_th_m]
        theta_d = params[n_th_m : n_th_m + n_th_d]
        delta = params[n_th_m + n_th_d :]
        ll = _joint_laplace_ll(
            theta_m,
            theta_d,
            delta,
            y - offset_arr,
            X,
            Z_m,
            W,
            Z_d,
            base_weights,
            lb_m,
            lb_d,
            warm,
        )
        return -float(ll)

    params0 = np.concatenate([theta_m0, theta_d0, delta0])

    # BCA's fixed point satisfies the conditional score equations, which
    # makes the joint Laplace gradient near zero in (β, v_m, v_d) but NOT
    # in (θ_m, θ_d, δ). L-BFGS-B with FD gradients can stall here on a
    # near-flat region, so we use Nelder-Mead (derivative-free, robust to
    # flat starts) to escape the BCA basin first, then polish with L-BFGS-B.
    nm = opt.minimize(
        outer_obj,
        params0,
        method="Nelder-Mead",
        options={
            "xatol": 1e-4,
            "fatol": 1e-5,
            "maxiter": 400 * params0.size,
            "adaptive": True,
        },
    )
    # Clip to bounds before L-BFGS-B (Nelder-Mead is unconstrained).
    polished0 = np.array(
        [
            min(
                max(x, lo if lo is not None else -np.inf),
                hi if hi is not None else np.inf,
            )
            for x, (lo, hi) in zip(nm.x, bounds, strict=True)
        ]
    )
    res = opt.minimize(
        outer_obj,
        polished0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-9, "gtol": 1e-7, "maxiter": 300},
    )
    # Keep the better of the two passes.
    if nm.fun < res.fun:
        res = nm

    theta_m_hat = res.x[:n_th_m]
    theta_d_hat = res.x[n_th_m : n_th_m + n_th_d]
    delta_hat = res.x[n_th_m + n_th_d :]
    final_ll = -res.fun

    # --- 4. Recover final state ---
    beta_hat = np.asarray(warm["beta"], dtype=np.float64)
    v_m_hat = (
        np.asarray(warm["v_m"], dtype=np.float64)
        if warm["v_m"] is not None
        else np.zeros(qm)
    )
    v_d_hat = (
        np.asarray(warm["v_d"], dtype=np.float64)
        if warm["v_d"] is not None
        else np.zeros(qd)
    )
    Lambda_m = lb_m.update(theta_m_hat)
    Lambda_d = lb_d.update(theta_d_hat)
    if qm:
        b_hat = np.asarray(Lambda_m @ v_m_hat).squeeze()
        Zb_hat = np.asarray(Z_m @ b_hat).squeeze()
    else:
        b_hat = np.zeros(0)
        Zb_hat = np.zeros(n)
    if qd:
        u_d_hat = np.asarray(Lambda_d @ v_d_hat).squeeze()
        Zd_ud_hat = np.asarray(Z_d @ u_d_hat).squeeze()
    else:
        u_d_hat = np.zeros(0)
        Zd_ud_hat = np.zeros(n)
    log_sigma = W @ delta_hat + Zd_ud_hat
    sigma_sq = np.exp(2.0 * log_sigma)
    fittedvalues = X @ beta_hat + Zb_hat + offset_arr
    resid_full = y - fittedvalues

    # --- 5. SEs via Hessian inverse for β at the Gaussian-heteroscedastic optimum ---
    # H_bb = X' diag(1/σ²) X. SEs = sqrt(diag(H_bb⁻¹)).
    inv_sigma_sq = 1.0 / sigma_sq
    H_bb = X.T @ (base_weights[:, None] * inv_sigma_sq[:, None] * X)
    try:
        fe_cov = np.linalg.inv(H_bb)
    except np.linalg.LinAlgError:
        fe_cov = np.linalg.pinv(H_bb)
    fe_bse_arr = np.sqrt(np.maximum(np.diag(fe_cov), 0.0))

    # --- 6. Variance components ---
    vc_m = _extract_vc(theta_m_hat, specs_m)
    vc_d = _extract_vc(theta_d_hat, disp_re_specs)

    # --- 7. BLUPs per spec ---
    mean_re = _split_blups(b_hat, specs_m, nlev_m, prefix="")
    disp_re = _split_blups(u_d_hat, disp_re_specs, nlev_d, prefix="")

    # --- 8. Package ---
    import scipy.stats as stats

    z_scores = beta_hat / np.where(fe_bse_arr > 0, fe_bse_arr, 1.0)
    fe_df_arr = np.full(p, float(max(n - p, 1)))
    fe_pvalues_arr = 2.0 * (1.0 - stats.t.cdf(np.abs(z_scores), df=fe_df_arr))
    fe_params = pd.Series(beta_hat, index=term_names)
    fe_bse = pd.Series(fe_bse_arr, index=term_names)
    fe_pvalues = pd.Series(fe_pvalues_arr, index=term_names)
    fe_df = pd.Series(fe_df_arr, index=term_names)
    fe_conf_int = pd.DataFrame(
        {
            "lower": beta_hat - 1.96 * fe_bse_arr,
            "upper": beta_hat + 1.96 * fe_bse_arr,
        },
        index=term_names,
    )

    endog_name = formula.split("~", 1)[0].strip()
    model = ModelInfo(
        exog=X,
        endog=y,
        groups=nw_data[specs_m[0].group].to_numpy(),
        endog_names=endog_name,
        formula=formula,
        data=_DataWrapper(frame=data),
    )

    nparams = p + n_th_m + n_th_d + p_d
    aic = -2.0 * final_ll + 2.0 * nparams
    bic = -2.0 * final_ll + np.log(n) * nparams

    ngroups_m = {s.group: nlev for s, nlev in zip(specs_m, nlev_m, strict=True)}

    return CrossedLMEResult(
        fe_params=fe_params,
        fe_bse=fe_bse,
        fe_pvalues=fe_pvalues,
        fe_conf_int=fe_conf_int,
        fe_df=fe_df,
        random_effects=mean_re,
        variance_components=vc_m,
        theta=np.concatenate([theta_m_hat, theta_d_hat]),
        resid=resid_full,
        fittedvalues=fittedvalues,
        scale=1.0,
        fe_cov=fe_cov,
        model=model,
        converged=bool(res.success),
        nobs=n,
        ngroups=ngroups_m,
        method=method,
        llf=float(final_ll),
        aic=float(aic),
        bic=float(bic),
        nparams=nparams,
        _primary_group_col=specs_m[0].group,
        _random_specs=list(specs_m),
        df_method=df_method,
        disp_params=pd.Series(delta_hat, index=disp_term_names),
        disp_random_effects=disp_re,
        disp_variance_components=vc_d,
        dispersion=sigma_sq,
        disp_method="joint_laplace",
    )


def _joint_laplace_ll(
    theta_m: np.ndarray,
    theta_d: np.ndarray,
    delta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z_m: sp.csc_matrix,
    W: np.ndarray,
    Z_d: sp.csc_matrix,
    weights: np.ndarray,
    lb_m: LambdaBuilder,
    lb_d: LambdaBuilder,
    warm: dict[str, np.ndarray | None],
) -> float:
    """Evaluate the joint Laplace marginal log-lik at (θ_m, θ_d, δ).

    Side effect: updates ``warm`` with the inner-mode (β, v_m, v_d).
    """
    Lambda_m = lb_m.update(theta_m)
    Lambda_d = lb_d.update(theta_d)
    qm = int(Lambda_m.shape[1])
    qd = int(Lambda_d.shape[1])
    n, p = X.shape

    # Initialize state
    beta = warm["beta"] if warm["beta"] is not None else np.zeros(p)
    v_m = warm["v_m"] if warm["v_m"] is not None else np.zeros(qm)
    v_d = warm["v_d"] if warm["v_d"] is not None else np.zeros(qd)
    beta = np.asarray(beta, dtype=np.float64).copy()
    v_m = np.asarray(v_m, dtype=np.float64).copy()
    v_d = np.asarray(v_d, dtype=np.float64).copy()

    Z_star_m = (Z_m @ Lambda_m).tocsc()
    Z_star_d = (Z_d @ Lambda_d).tocsc()

    # Inner mode-finding: alternate exact (β, v_m) update and v_d Newton step.
    for _ in range(_INNER_MAXITER):
        # --- (i) Compute current σ²
        if qd:
            u_d = np.asarray(Lambda_d @ v_d).squeeze()
            Zd_u_d = np.asarray(Z_d @ u_d).squeeze()
        else:
            Zd_u_d = np.zeros(n)
        log_sigma = W @ delta + Zd_u_d
        sigma_sq = np.exp(2.0 * log_sigma)
        inv_sigma_sq = 1.0 / sigma_sq
        w_eff = weights * inv_sigma_sq  # (n,)

        # --- (ii) Exact (β, v_m) solve given σ² (Gaussian heteroscedastic) ---
        beta_new, v_m_new = _solve_beta_vm(
            y,
            X,
            Z_star_m,
            w_eff,
            qm,
        )

        # --- (iii) v_d Newton step given (β_new, v_m_new) ---
        if qm:
            b_new = np.asarray(Lambda_m @ v_m_new).squeeze()
            Zb_new = np.asarray(Z_m @ b_new).squeeze()
        else:
            Zb_new = np.zeros(n)
        e = y - X @ beta_new - Zb_new
        e_sq_over_sigma_sq = (e * e) * inv_sigma_sq

        score_vd = (
            np.asarray(Z_star_d.T @ (weights * (e_sq_over_sigma_sq - 1.0))).squeeze()
            - v_d
        )
        # Hessian for v_d Newton (block-diagonal Laplace approximation):
        #   H_dd = 2 * Z*_d' diag(w * e²/σ²) Z*_d + I
        w_dd = weights * e_sq_over_sigma_sq  # (n,)
        H_dd = _build_H_dd(Z_star_d, w_dd, qd)

        # Newton step with step-halving for stability.
        try:
            step_vd = spla.spsolve(H_dd, score_vd) if qd > 0 else np.zeros(0)
        except Exception:
            step_vd = score_vd / (2.0 * np.sum(weights * e_sq_over_sigma_sq) + 1.0)

        # Step-halving cap.
        step_norm = float(np.max(np.abs(step_vd))) if step_vd.size else 0.0
        scale = 1.0 if step_norm < 4.0 else 4.0 / max(step_norm, 1e-12)
        v_d_new = v_d + scale * step_vd

        # Convergence check on all blocks.
        d_beta = np.max(np.abs(beta_new - beta)) if p > 0 else 0.0
        d_vm = np.max(np.abs(v_m_new - v_m)) if qm > 0 else 0.0
        d_vd = np.max(np.abs(v_d_new - v_d)) if qd > 0 else 0.0

        beta = beta_new
        v_m = v_m_new
        v_d = v_d_new
        if max(d_beta, d_vm, d_vd) < _INNER_TOL:
            break

    warm["beta"] = beta
    warm["v_m"] = v_m
    warm["v_d"] = v_d

    # ---- Final marginal log-lik ----
    u_d = np.asarray(Lambda_d @ v_d).squeeze() if qd else np.zeros(qd)
    Zd_u_d = np.asarray(Z_d @ u_d).squeeze() if qd else np.zeros(n)
    log_sigma = W @ delta + Zd_u_d
    sigma_sq = np.exp(2.0 * log_sigma)
    inv_sigma_sq = 1.0 / sigma_sq
    if qm:
        b = np.asarray(Lambda_m @ v_m).squeeze()
        Zb = np.asarray(Z_m @ b).squeeze()
    else:
        Zb = np.zeros(n)
    e = y - X @ beta - Zb
    e_sq_over_sigma_sq = (e * e) * inv_sigma_sq

    # Conditional ll: -0.5 * Σ w_i [log(2π) + 2 log σ_i + e²/σ²]
    cond_ll = -0.5 * float(
        np.sum(weights * (np.log(2.0 * np.pi) + np.log(sigma_sq) + e_sq_over_sigma_sq))
    )
    penalty = 0.5 * float(np.dot(v_m, v_m) + np.dot(v_d, v_d))

    # log|H_mm| where H_mm = Z*_m' diag(w * 1/σ²) Z*_m + I_qm
    w_eff = weights * inv_sigma_sq
    H_mm = _build_H_mm(Z_star_m, w_eff, qm)
    log_det_Hmm = _sparse_chol_logdet(H_mm) if qm else 0.0

    w_dd_final = weights * e_sq_over_sigma_sq
    H_dd_final = _build_H_dd(Z_star_d, w_dd_final, qd)
    log_det_Hdd = _sparse_chol_logdet(H_dd_final) if qd else 0.0

    laplace_correction = 0.5 * (log_det_Hmm + log_det_Hdd)
    ll = cond_ll - penalty - laplace_correction
    return float(ll)


def _solve_beta_vm(
    y: np.ndarray,
    X: np.ndarray,
    Z_star_m: sp.csc_matrix,
    w_eff: np.ndarray,
    qm: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact joint solve for (β, v_m) at fixed σ² (Gaussian heteroscedastic).

    Minimises  ½ Σ w_i (y_i − X_i β − Z*_m,i v_m)²  + ½ ‖v_m‖²,
    which is the conditional posterior mode for (β, v_m) given v_d.
    """
    n, p = X.shape
    sqrt_w = np.sqrt(w_eff)
    Xw = sqrt_w[:, None] * X
    yw = sqrt_w * y
    if qm == 0:
        # Plain weighted OLS
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        try:
            beta = la.solve(XtX, Xty, assume_a="pos")
        except la.LinAlgError:
            beta = la.lstsq(XtX, Xty)[0]
        return beta, np.zeros(0)

    # Schur complement: solve for β via marginalising v_m.
    Zw = _scale_rows_csc(Z_star_m, sqrt_w)  # (n, qm) sparse
    XtX = Xw.T @ Xw  # (p, p) dense
    XtZ = np.asarray((Xw.T @ Zw).toarray() if sp.issparse(Xw.T @ Zw) else Xw.T @ Zw)
    ZtZ = (Zw.T @ Zw + sp.eye(qm, format="csc")).tocsc()
    Xty = Xw.T @ yw  # (p,)
    Zty = np.asarray(Zw.T @ yw).squeeze()  # (qm,)

    # Solve [XtX  XtZ ] [β  ] = [Xty]
    #       [XtZ' ZtZ] [v_m]   [Zty]
    # via Schur on the (β) block. Standard LMM Henderson formula.
    try:
        ZtZ_inv_XtZt = np.column_stack(
            [spla.spsolve(ZtZ, XtZ[:, j]) for j in range(p)]
        )  # (qm, p)
        ZtZ_inv_Zty = spla.spsolve(ZtZ, Zty)  # (qm,)
    except Exception:
        ZtZ_dense = ZtZ.toarray()
        ZtZ_inv_XtZt = la.solve(ZtZ_dense, XtZ.T).T  # wait shape...
        # Actually XtZ shape (p, qm), so ZtZ_inv @ XtZ.T → (qm, p)
        ZtZ_inv_XtZt = la.solve(ZtZ_dense, XtZ.T)  # (qm, p)
        ZtZ_inv_Zty = la.solve(ZtZ_dense, Zty)

    Schur = XtX - XtZ @ ZtZ_inv_XtZt  # (p, p)
    rhs_beta = Xty - XtZ @ ZtZ_inv_Zty
    try:
        beta = la.solve(Schur, rhs_beta, assume_a="pos")
    except la.LinAlgError:
        beta = la.lstsq(Schur, rhs_beta)[0]

    v_m = ZtZ_inv_Zty - ZtZ_inv_XtZt @ beta
    return beta, v_m


def _scale_rows_csc(M: sp.csc_matrix, s: np.ndarray) -> sp.csc_matrix:
    """Return diag(s) @ M for sparse CSC M (row-scaling)."""
    if not sp.issparse(M):
        return s[:, None] * M
    coo = M.tocoo()
    new_data = coo.data * s[coo.row]
    return sp.csc_matrix((new_data, (coo.row, coo.col)), shape=coo.shape)


def _build_H_mm(Z_star_m: sp.csc_matrix, w_eff: np.ndarray, qm: int) -> sp.csc_matrix:
    if qm == 0:
        return sp.csc_matrix((0, 0))
    sqrt_w = np.sqrt(w_eff)
    Zw = _scale_rows_csc(Z_star_m, sqrt_w)
    return (Zw.T @ Zw + sp.eye(qm, format="csc")).tocsc()


def _build_H_dd(Z_star_d: sp.csc_matrix, w_dd: np.ndarray, qd: int) -> sp.csc_matrix:
    if qd == 0:
        return sp.csc_matrix((0, 0))
    sqrt_w = np.sqrt(np.maximum(w_dd, 0.0))
    Zw = _scale_rows_csc(Z_star_d, sqrt_w)
    return (2.0 * (Zw.T @ Zw) + sp.eye(qd, format="csc")).tocsc()


def _sparse_chol_logdet(A: sp.csc_matrix) -> float:
    """log|A| for a sparse SPD CSC matrix.  Falls back to a dense
    Cholesky if sksparse isn't installed (sufficient for the small q
    sizes typical of dispformula side)."""
    try:
        from sksparse.cholmod import (  # type: ignore[import-untyped,import-not-found,unused-ignore]  # noqa: I001, E501
            cholesky as cholmod_cholesky,
        )

        return float(cholmod_cholesky(A).logdet())
    except Exception:
        try:
            L = la.cholesky(A.toarray(), lower=True)
            return float(2.0 * np.sum(np.log(np.diag(L))))
        except la.LinAlgError:
            sign, logdet = np.linalg.slogdet(A.toarray())
            return float(logdet) if sign > 0 else 1e300


def _extract_vc(theta: np.ndarray, specs: list[Any]) -> dict[str, Any]:
    """Translate theta → variance components per spec.

    Assumes scalar/diagonal specs (no random slopes).  For correlated
    multi-term specs, returns DataFrames as per the LMM convention.
    """
    vc: dict[str, Any] = {}
    idx = 0
    for spec in specs:
        if spec.n_terms == 1:
            vc[spec.group] = float(theta[idx] ** 2)
            idx += 1
        elif spec.correlated:
            # Lower-triangular packing of L; rebuild L and L L'
            p_j = spec.n_terms
            L = np.zeros((p_j, p_j))
            for r in range(p_j):
                for c in range(r + 1):
                    L[r, c] = theta[idx]
                    idx += 1
            cov = L @ L.T
            vc[spec.group] = pd.DataFrame(cov)
        else:
            # Independent: theta[idx:idx+p_j] are SDs
            p_j = spec.n_terms
            cov = np.diag(theta[idx : idx + p_j] ** 2)
            vc[spec.group] = pd.DataFrame(cov)
            idx += p_j
    return vc


def _split_blups(
    blups: np.ndarray,
    specs: list[Any],
    n_levels: list[int],
    prefix: str = "",
) -> dict[str, Any]:
    """Slice a stacked BLUP vector into per-spec Series/DataFrames."""
    out: dict[str, Any] = {}
    offset = 0
    for spec, q_j in zip(specs, n_levels, strict=True):
        n_blups_j = spec.n_terms * q_j
        block = blups[offset : offset + n_blups_j]
        if spec.n_terms == 1:
            out[f"{prefix}{spec.group}"] = pd.Series(block)
        else:
            out[f"{prefix}{spec.group}"] = pd.DataFrame(
                block.reshape(q_j, spec.n_terms)
            )
        offset += n_blups_j
    return out
