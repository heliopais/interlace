"""interlace: REML estimation for linear mixed models with crossed random intercepts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.stats as stats

from interlace.augment import hlm_augment
from interlace.formula import (
    groups_to_random_effects,
    parse_formula,
    parse_random_effects,
)
from interlace.influence import (
    cooks_distance,
    hlm_influence,
    mdffits,
    n_influential,
    tau_gap,
)
from interlace.leverage import leverage
from interlace.plotting import dotplot_diag, plot_influence, plot_resid
from interlace.profiled_reml import (
    _build_A11,
    _precompute,
    _sparse_solve,
    fit_reml,
    make_lambda,
)
from interlace.residuals import hlm_resid
from interlace.result import CrossedLMEResult, ModelInfo, _DataWrapper
from interlace.sparse_z import build_joint_z_from_specs

__all__ = [
    "fit",
    "CrossedLMEResult",
    # Residuals
    "hlm_resid",
    # Leverage
    "leverage",
    # Influence diagnostics
    "hlm_influence",
    "cooks_distance",
    "mdffits",
    "n_influential",
    "tau_gap",
    # Combined
    "hlm_augment",
    # Plotting
    "plot_resid",
    "plot_influence",
    "dotplot_diag",
]


def fit(
    formula: str,
    data: pd.DataFrame,
    groups: str | list[str] | None = None,
    method: str = "REML",
    random: list[str] | None = None,
) -> CrossedLMEResult:
    """Fit a linear mixed model with crossed random effects via profiled REML.

    Parameters
    ----------
    formula:
        Patsy-syntax fixed-effects formula, e.g. ``"y ~ x1 + x2"``.
    data:
        DataFrame containing all variables.
    groups:
        Column name (str) or list of column names for crossed random
        intercepts. Shorthand for ``random=["(1|g1)", "(1|g2)", ...]``.
        The first entry is the primary grouping factor.
    random:
        List of lme4-style random effect specifications, e.g.
        ``["(1 + x | g1)", "(1 | g2)"]``.  Supports correlated (``|``) and
        independent (``||``) parameterisations.  Takes precedence over
        *groups* when both are provided.
    method:
        Estimation method; only ``"REML"`` is currently supported.

    Returns
    -------
    CrossedLMEResult
        Drop-in replacement for statsmodels ``MixedLMResults``.
    """
    if method != "REML":
        raise ValueError(f"Only method='REML' is supported; got '{method}'")
    if random is None and groups is None:
        raise ValueError("Either 'groups' or 'random' must be provided.")

    # --- Build RandomEffectSpec list ---
    if random is not None:
        specs = parse_random_effects(random)
    else:
        specs = groups_to_random_effects(groups)  # type: ignore[arg-type]

    group_cols = [s.group for s in specs]

    # --- 1. Parse fixed-effects formula ---
    parsed = parse_formula(formula, data, groups=group_cols[0])
    y = parsed.y
    X = parsed.X
    term_names = parsed.term_names
    n, p = X.shape

    # --- 2. Build joint sparse Z and collect n_levels per spec ---
    Z = build_joint_z_from_specs(specs, data)
    n_levels_list: list[int] = []
    for spec in specs:
        _codes, _uniques = pd.factorize(data[spec.group], sort=True)
        n_levels_list.append(len(_uniques))

    # --- 3. Fit REML ---
    reml = fit_reml(y, X, Z, q_sizes=[], specs=specs, n_levels=n_levels_list)

    # --- 4. Recover quantities at optimum ---
    Lambda = make_lambda(reml.theta, specs, n_levels_list)
    cache = _precompute(y, X, Z)

    ZtZ = sp.csc_matrix(cache["ZtZ"])
    ZtX = np.asarray(cache["ZtX"])
    Zty = np.asarray(cache["Zty"])
    XtX = np.asarray(cache["XtX"])
    Xty = np.asarray(cache["Xty"])

    A11 = _build_A11(ZtZ, Lambda)
    lZty = np.asarray(Lambda.T @ Zty).squeeze()
    lZtX = np.asarray(Lambda.T @ ZtX)
    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X  # X'Ω⁻¹X  (p×p)
    rhs = Xty - lZtX.T @ c1  # X'Ω⁻¹y  (p,)
    beta = la.solve(MX, rhs, assume_a="pos")

    # --- 5. BLUPs: b̂ = Lambda A11⁻¹ Lambda' Z'ε ---
    eps = y - X @ beta
    Zte = np.asarray(Z.T @ eps).squeeze()
    blups = np.asarray(
        Lambda @ _sparse_solve(A11, np.asarray(Lambda.T @ Zte).squeeze())
    ).squeeze()

    # --- 6. Fitted values and conditional residuals ---
    fittedvalues = X @ beta + Z @ blups
    resid = y - fittedvalues

    # --- 7. Standard errors (Wald) ---
    sigma2 = reml.sigma2
    MX_inv = np.linalg.inv(MX)
    fe_cov = sigma2 * MX_inv
    fe_bse = np.sqrt(sigma2 * np.diag(MX_inv))
    z_scores = beta / fe_bse
    fe_pvalues = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))
    fe_conf_int = pd.DataFrame(
        {"lower": beta - 1.96 * fe_bse, "upper": beta + 1.96 * fe_bse},
        index=term_names,
    )

    # --- 8. Package random effects per spec ---
    # For intercept-only specs (n_terms == 1): Series of q BLUPs (backward compat).
    # For multi-term specs: flat Series of n_terms*q BLUPs.
    # NOTE: interlace-hnl will upgrade multi-term to DataFrame + covariance matrix.
    random_effects: dict[str, pd.Series] = {}
    variance_components: dict[str, float] = {}
    ngroups: dict[str, int] = {}
    theta_idx = 0
    blup_offset = 0
    for spec, q_j in zip(specs, n_levels_list, strict=True):
        from interlace.profiled_reml import n_theta_for_spec

        n_theta_j = n_theta_for_spec(spec.n_terms, spec.correlated)
        n_blups_j = spec.n_terms * q_j
        blup_block = blups[blup_offset : blup_offset + n_blups_j]
        uniques: list[object] = sorted(data[spec.group].unique())

        if spec.n_terms == 1:
            # Intercept-only: backward-compatible Series
            re_series = pd.Series(blup_block, index=uniques, name=spec.group)
            theta_j0 = reml.theta[theta_idx]
            vc_val = float(sigma2 * theta_j0**2)
        else:
            # Multi-term: flat Series for now (interlace-hnl will upgrade)
            re_series = pd.Series(blup_block, name=spec.group)
            # Variance component: variance of intercept term (theta_j[0]^2 * sigma2)
            theta_j0 = reml.theta[theta_idx]
            vc_val = float(sigma2 * theta_j0**2)

        random_effects[spec.group] = re_series
        variance_components[spec.group] = vc_val
        ngroups[spec.group] = q_j
        theta_idx += n_theta_j
        blup_offset += n_blups_j

    # --- 9. Build ModelInfo ---
    model_info = ModelInfo(
        exog=X,
        endog=y,
        groups=np.asarray(data[group_cols[0]]),
        endog_names=formula.split("~")[0].strip(),
        formula=formula,
        data=_DataWrapper(frame=data),
    )

    return CrossedLMEResult(
        fe_params=pd.Series(beta, index=term_names),
        fe_bse=pd.Series(fe_bse, index=term_names),
        fe_pvalues=pd.Series(fe_pvalues, index=term_names),
        fe_conf_int=fe_conf_int,
        random_effects=random_effects,
        variance_components=variance_components,
        theta=reml.theta,
        resid=np.asarray(resid),
        fittedvalues=np.asarray(fittedvalues),
        scale=sigma2,
        fe_cov=fe_cov,
        model=model_info,
        converged=reml.converged,
        nobs=n,
        ngroups=ngroups,
        method=method,
        llf=reml.llf,
        aic=reml.aic,
        bic=reml.bic,
        _gpgap_group_col=group_cols[0],
        _gpgap_vc_cols=group_cols[1:],
    )
